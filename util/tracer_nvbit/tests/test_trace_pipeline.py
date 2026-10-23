import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

TRACER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACER))
from trace_pipeline import TracePipeline, run_phase


class TracePipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def script(self, directory, name, body):
        directory.mkdir(parents=True, exist_ok=True)
        command = shlex.join([sys.executable, "-c", body])
        (directory / name).write_text("set -e\n" + command + "\n")

    def case(self, name, capture="pass", cpu="pass"):
        directory = self.root / name
        self.script(directory, "run_capture.sh", capture)
        self.script(directory, "run_postprocess.sh", cpu)
        return directory

    def test_next_capture_runs_before_prior_cpu_completion(self):
        marker = self.root / "capture-two"
        started = self.root / "cpu-one-started"
        first = self.case("one", cpu=f"""
import pathlib, time
pathlib.Path({str(started)!r}).touch()
deadline = time.monotonic() + 5
while not pathlib.Path({str(marker)!r}).exists():
    if time.monotonic() > deadline:
        raise RuntimeError('capture two did not overlap CPU one')
    time.sleep(0.01)
""")
        second = self.case("two", capture=f"""
import pathlib, time
deadline = time.monotonic() + 5
while not pathlib.Path({str(started)!r}).exists():
    if time.monotonic() > deadline:
        raise RuntimeError('CPU one did not start')
    time.sleep(0.01)
pathlib.Path({str(marker)!r}).touch()
""")
        with TracePipeline(workers=1, max_pending=2) as pipeline:
            pipeline.capture(first)
            pipeline.capture(second)
        cpu = json.loads((first / "postprocess.receipt.json").read_text())
        capture = json.loads((second / "capture.receipt.json").read_text())
        self.assertLess(capture["started_monotonic_ns"], cpu["ended_monotonic_ns"])
        self.assertEqual(cpu["exit_code"], 0)

    def test_queue_depth_one_waits_for_cpu_before_next_capture(self):
        marker = self.root / "cpu-finished"
        first = self.case("one", cpu=f"import pathlib; pathlib.Path({str(marker)!r}).touch()")
        second = self.case("two", capture=f"import pathlib; assert pathlib.Path({str(marker)!r}).exists()")
        with TracePipeline(workers=1, max_pending=1) as pipeline:
            pipeline.capture(first)
            pipeline.capture(second)
        cpu = json.loads((first / "postprocess.receipt.json").read_text())
        capture = json.loads((second / "capture.receipt.json").read_text())
        self.assertLessEqual(cpu["ended_monotonic_ns"], capture["started_monotonic_ns"])

    def test_zero_workers_preserves_serial_order(self):
        first = self.case("one")
        second = self.case("two")
        with TracePipeline() as pipeline:
            pipeline.capture(first)
            pipeline.capture(second)
        cpu = json.loads((first / "postprocess.receipt.json").read_text())
        capture = json.loads((second / "capture.receipt.json").read_text())
        self.assertLessEqual(cpu["ended_monotonic_ns"], capture["started_monotonic_ns"])

    def test_cpu_has_no_device_or_cuda_injection(self):
        directory = self.case("CPU only", cpu="""
import os
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert 'CUDA_INJECTION64_PATH' not in os.environ
assert 'LD_PRELOAD' not in os.environ
""")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "test-device", "CUDA_INJECTION64_PATH": "test-injection", "LD_PRELOAD": ""}):
            run_phase(directory, "run_postprocess.sh", "postprocess")

    def test_capture_failure_keeps_evidence_and_skips_cpu(self):
        directory = self.case("failed-capture", capture="from pathlib import Path; Path('raw-evidence').write_text('partial'); raise SystemExit(7)", cpu="raise RuntimeError('CPU must not run')")
        with self.assertRaises(subprocess.CalledProcessError):
            with TracePipeline(workers=1) as pipeline:
                pipeline.capture(directory)
        self.assertEqual((directory / "raw-evidence").read_text(), "partial")
        self.assertFalse((directory / "postprocess.receipt.json").exists())
        self.assertEqual(json.loads((directory / "capture.receipt.json").read_text())["exit_code"], 7)

    def test_cpu_failure_propagates(self):
        directory = self.case("failed-cpu", cpu="raise SystemExit(9)")
        with self.assertRaises(subprocess.CalledProcessError):
            with TracePipeline(workers=1) as pipeline:
                pipeline.capture(directory)
        self.assertEqual(json.loads((directory / "postprocess.receipt.json").read_text())["exit_code"], 9)

    def test_repeated_directory_waits_before_reusing_files(self):
        directory = self.case("same-directory", cpu="from pathlib import Path; Path('CPU-complete').touch()")
        with TracePipeline(workers=1) as pipeline:
            pipeline.capture(directory)
            pipeline.wait_directory(directory)
            self.assertTrue((directory / "CPU-complete").exists())
            self.assertNotIn(str(directory.resolve()), pipeline.pending)

    def test_launcher_norun_writes_separate_scripts(self):
        common = types.ModuleType("common")
        binary = self.root / "fixture"
        binary.touch()
        common.load_defined_yamls = lambda: None
        common.gen_apps_from_suite_list = lambda suites: [(str(self.root), str(self.root), "fixture", [{"args": "32 4", "name": "input"}])]
        common.get_cuda_version = lambda directory: "12.9"
        common.get_argfoldername = lambda argpair: argpair["name"]
        common.dir_option_test = lambda name, default, directory: name
        common.file_option_test = lambda name, default, directory: name
        common.PathMissing = FileNotFoundError
        output = self.root / "output with spaces"
        argv = [str(TRACER / "run_hw_trace.py"), "--norun", "--spinlock_handling", "none", "--postprocess-workers", "1", "--results-dir", str(output), "--keep-raw", "--limit_kernel_number", "4"]
        with patch.dict(sys.modules, {"common": common}), patch.object(sys, "argv", argv):
            runpy.run_path(str(TRACER / "run_hw_trace.py"), run_name="__main__")
        directory = output / "fixture/input"
        self.assertEqual((directory / "run.sh").read_text(), "set -e\nbash run_capture.sh\nbash run_postprocess.sh\n")
        self.assertIn('DYNAMIC_KERNEL_RANGE="0-4"', (directory / "run_capture.sh").read_text())
        self.assertNotIn("rm -f", (directory / "run_postprocess.sh").read_text())
        self.assertIn("CUDA_VISIBLE_DEVICES=", (directory / "run_postprocess.sh").read_text())
        self.assertFalse((directory / "capture.receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
