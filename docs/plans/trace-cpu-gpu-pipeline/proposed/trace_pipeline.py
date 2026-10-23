"""Run one GPU capture stream alongside bounded CPU postprocessing workers."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def run_phase(directory, script, phase):
    """Execute a generated phase script and preserve its output and host timing."""
    directory = Path(directory)
    environment = os.environ.copy()
    if phase == "postprocess":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment.pop("CUDA_INJECTION64_PATH", None)
        environment.pop("LD_PRELOAD", None)
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.monotonic_ns()
    argv = ["bash", str(directory / script)]
    with (directory / (phase + ".stdout.log")).open("w") as stdout, (
        directory / (phase + ".stderr.log")
    ).open("w") as stderr:
        try:
            result = subprocess.run(
                argv, cwd=directory, env=environment, stdout=stdout, stderr=stderr
            )
            code = result.returncode
        except OSError as error:
            stderr.write(str(error) + "\n")
            code = 127
    ended_ns = time.monotonic_ns()
    receipt = {
        "argv": argv,
        "cwd": str(directory),
        "phase": phase,
        "started_utc": started_utc,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "wall_seconds": (ended_ns - started_ns) / 1e9,
        "exit_code": code,
    }
    (directory / (phase + ".receipt.json")).write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return receipt


class TracePipeline:
    """The caller owns the GPU; workers only run completed bundles on the CPU."""

    def __init__(self, workers=0, max_pending=None):
        if workers < 0:
            raise ValueError("postprocess workers must be nonnegative")
        if max_pending is not None and max_pending < 1:
            raise ValueError("max pending postprocess must be positive")
        self.workers = workers
        self.max_pending = max_pending if max_pending is not None else max(1, 2 * workers)
        self.pending = {}
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers else None

    def __enter__(self):
        return self

    def reap(self, block=False):
        if block and self.pending:
            wait(self.pending.values(), return_when=FIRST_COMPLETED)
        for directory, future in list(self.pending.items()):
            if future.done():
                del self.pending[directory]
                future.result()

    def wait_directory(self, directory):
        directory = str(Path(directory).resolve())
        future = self.pending.pop(directory, None)
        if future is not None:
            future.result()

    def capture(self, directory, spinlock=False):
        directory = str(Path(directory).resolve())
        self.wait_directory(directory)
        self.reap()
        while len(self.pending) >= self.max_pending:
            self.reap(block=True)
        if spinlock:
            run_phase(directory, "run_spinlock_detection.sh", "spinlock")
        run_phase(directory, "run_capture.sh", "capture")
        if self.pool is None:
            run_phase(directory, "run_postprocess.sh", "postprocess")
        else:
            self.pending[directory] = self.pool.submit(
                run_phase, directory, "run_postprocess.sh", "postprocess"
            )

    def __exit__(self, exception_type, exception, traceback):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        failures = []
        for directory, future in self.pending.items():
            try:
                future.result()
            except Exception as error:
                failures.append((directory, error))
        if failures:
            for directory, error in failures:
                print("CPU postprocess failed in {}: {}".format(directory, error))
            if exception is None:
                raise failures[0][1]
        return False
