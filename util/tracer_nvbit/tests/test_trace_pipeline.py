import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

TRACER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACER))
spec = importlib.util.spec_from_file_location("two_phase_launcher", TRACER / "run_hw_trace.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TracePhasesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.here = self.root / "checkout/util/tracer_nvbit"
        (self.here / "tracer_tool/formatting").mkdir(parents=True)
        (self.here / "tracer_tool/traces-processing").mkdir()
        (self.here / "tracer_tool/tracer_tool.so").write_bytes(b"test tracer identity")
        self.applications = self.root / "applications"
        self.applications.mkdir()
        self.executable(self.applications / "fixture", """
+from pathlib import Path
+import os
+root = Path(os.environ['TRACES_FOLDER']) / 'traces'
+(root / 'kernelslist_ctx_fixture').write_text('kernel-1.trace.xz\\n')
+(root / 'kernel-1.trace.records').write_bytes(b'unit workflow record data')
+""".replace("\n+", "\n"))
        self.executable(self.here / "tracer_tool/formatting/format-trace-capture", """
+import os, time
+from pathlib import Path
+assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
+assert 'CUDA_INJECTION64_PATH' not in os.environ
+assert 'LD_PRELOAD' not in os.environ
+if os.environ.get('TEST_CPU_STARTED'):
+    Path(os.environ['TEST_CPU_STARTED']).touch()
+    deadline = time.monotonic() + 5
+    while not Path(os.environ['TEST_CPU_RELEASE']).exists():
+        if time.monotonic() > deadline:
+            raise RuntimeError('CPU test gate was not released')
+        time.sleep(0.01)
+""".replace("\n+", "\n"))
        self.executable(self.here / "tracer_tool/traces-processing/post-traces-processing", "pass")
        self.common = types.ModuleType("common")
        self.common.load_defined_yamls = lambda: None
        self.common.gen_apps_from_suite_list = lambda suites: [(str(self.applications), str(self.applications), "fixture", [{"args": "", "name": "input"}])]
        self.common.get_cuda_version = lambda directory: "12.9"
        self.common.get_argfoldername = lambda row: row["name"]
        self.common.file_option_test = lambda path, default, directory: path
        self.common.dir_option_test = self.directory
        self.common.PathMissing = FileNotFoundError
        self.patch_here = patch.object(launcher, "HERE", self.here)
        self.patch_here.start(); self.addCleanup(self.patch_here.stop)
        self.patch_common = patch.dict(sys.modules, {"common": self.common})
        self.patch_common.start(); self.addCleanup(self.patch_common.stop)

    @staticmethod
    def directory(path, default, directory):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def executable(path, body):
        path.write_text("#!" + sys.executable + "\n" + body + "\n")
        path.chmod(0o755)

    def arguments(self, name):
        return argparse.Namespace(results_dir=self.root / name, traces_dir=self.root / (name + "-traces"), benchmark_list="fixture", device_num="test-device", kernel_number=-99, terminate_upon_limit=False, spinlock_handling="none", spinlock_fast_forward_iterations=1, postprocess_threads=2, postprocess_format="text", cpu_workers=1)

    def test_gpu_phase_never_constructs_or_joins_cpu_workers(self):
        args = self.arguments("GPU only")
        with patch.object(launcher, "ThreadPoolExecutor", side_effect=AssertionError("GPU phase invoked CPU work")):
            self.assertEqual(launcher.gpu_phase(args), 0)
        case = args.results_dir / "fixture/input"
        self.assertTrue((case / "gpu.receipt.json").exists())
        self.assertFalse((case / "run_cpu.sh").exists())
        self.assertFalse((args.results_dir / "cpu-traces.json").exists())
        script = (case / "run_gpu.sh").read_text()
        self.assertIn("TRACE_DEFER_FORMATTING=1", script)
        self.assertNotIn("format-trace-capture", script)

    def test_cpu_phase_uses_saved_gpu_manifest_without_cuda_discovery(self):
        args = self.arguments("cpu-input")
        self.assertEqual(launcher.gpu_phase(args), 0)
        self.common.get_cuda_version = lambda directory: self.fail("CPU phase asked for CUDA")
        self.common.gen_apps_from_suite_list = lambda suites: self.fail("CPU phase discovered apps")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "test-device", "CUDA_INJECTION64_PATH": "test-only", "LD_PRELOAD": ""}):
            self.assertEqual(launcher.cpu_phase(args), 0)
        self.assertEqual(json.loads((args.results_dir / "cpu-traces.json").read_text())["cases"][0]["status"], "processed")

    def test_next_gpu_phase_completes_while_cpu_phase_is_held(self):
        first = self.arguments("first")
        second = self.arguments("second")
        self.assertEqual(launcher.gpu_phase(first), 0)
        started = self.root / "CPU-started"
        release = self.root / "CPU-release"
        with patch.dict(os.environ, {"TEST_CPU_STARTED": str(started), "TEST_CPU_RELEASE": str(release)}):
            with ThreadPoolExecutor(max_workers=1) as pool:
                worker = pool.submit(launcher.cpu_phase, first)
                try:
                    deadline = time.monotonic() + 5
                    while not started.exists():
                        if time.monotonic() > deadline:
                            self.fail("Separate CPU process did not start")
                        time.sleep(0.01)
                    self.assertEqual(launcher.gpu_phase(second), 0)
                    self.assertFalse(worker.done())
                    self.assertTrue((second.results_dir / "gpu-traces.json").exists())
                finally:
                    release.touch()
                self.assertEqual(worker.result(), 0)

    def test_cpu_failure_does_not_change_collected_gpu_status(self):
        args = self.arguments("CPU failure")
        self.assertEqual(launcher.gpu_phase(args), 0)
        before = (args.results_dir / "gpu-traces.json").read_bytes()
        self.executable(self.here / "tracer_tool/formatting/format-trace-capture", "raise SystemExit(9)")
        self.assertEqual(launcher.cpu_phase(args), 1)
        self.assertEqual((args.results_dir / "gpu-traces.json").read_bytes(), before)
        self.assertTrue((args.traces_dir / "fixture/input/traces/kernel-1.trace.records").exists())

    def test_gpu_failure_is_recorded_and_not_given_to_cpu(self):
        args = self.arguments("GPU failure")
        self.executable(self.applications / "fixture", "raise SystemExit(7)")
        self.assertEqual(launcher.gpu_phase(args), 1)
        self.assertEqual(launcher.cpu_phase(args), 0)
        self.assertEqual(json.loads((args.results_dir / "gpu-traces.json").read_text())["cases"][0]["status"], "failed")
        self.assertEqual(json.loads((args.results_dir / "cpu-traces.json").read_text())["cases"], [])

    def test_existing_trace_directory_is_preserved(self):
        first = self.arguments("first-owner")
        self.assertEqual(launcher.gpu_phase(first), 0)
        record = first.traces_dir / "fixture/input/traces/kernel-1.trace.records"
        before = record.read_bytes()
        second = self.arguments("second-owner")
        second.traces_dir = first.traces_dir
        with self.assertRaises(FileExistsError):
            launcher.gpu_phase(second)
        self.assertEqual(record.read_bytes(), before)

    def test_existing_collection_is_preserved(self):
        args = self.arguments("retained")
        self.assertEqual(launcher.gpu_phase(args), 0)
        manifest = (args.results_dir / "gpu-traces.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "fresh GPU results"):
            launcher.gpu_phase(args)
        self.assertEqual((args.results_dir / "gpu-traces.json").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
