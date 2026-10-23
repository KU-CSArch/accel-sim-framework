"""Record the outcome of one explicitly selected GPU or CPU phase command."""

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
    if phase == "cpu":
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
