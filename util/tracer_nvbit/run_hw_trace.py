#!/usr/bin/env python3
"""Collect GPU records or process an existing collection on CPU-only workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from trace_pipeline import run_phase

HERE = Path(__file__).resolve().parent


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_phase(args):
    # CPU phase never imports application discovery or asks for a CUDA version.
    sys.path.insert(0, str(HERE.parent / "job_launching"))
    import common
    common.load_defined_yamls()
    benchmarks = common.gen_apps_from_suite_list(args.benchmark_list.split(","))
    cuda_version = common.get_cuda_version(str(HERE) + "/")
    root = args.results_dir.resolve()
    traces = (args.traces_dir or root / "collected-traces").resolve()
    manifest = root / "gpu-traces.json"
    if manifest.exists():
        raise RuntimeError("Select a fresh GPU results directory; a collection already exists")
    root.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    tracer = HERE / "tracer_tool/tracer_tool.so"
    result = {
        "phase": "gpu", "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device_num, "cuda_version": cuda_version,
        "tracer_sha256": digest(tracer), "traces_dir": str(traces),
        "cases": [], "completed": False,
    }
    save(manifest, result)
    for executable_dir, data_dir, executable, inputs in benchmarks:
        for input_row in inputs:
            name = str(Path(executable) / common.get_argfoldername(input_row))
            directory = root / name
            trace_root = traces / name
            trace_directory = trace_root / "traces"
            directory.mkdir(parents=True, exist_ok=False)
            trace_directory.mkdir(parents=True, exist_ok=False)
            data_dir = common.dir_option_test(data_dir, "", str(HERE))
            (directory / "data_dirs").symlink_to(data_dir, target_is_directory=True)
            try:
                selected = common.dir_option_test(str(Path(data_dir) / executable / "data"), "", str(HERE))
                (directory / "data").symlink_to(selected, target_is_directory=True)
            except common.PathMissing:
                pass
            executable_path = common.file_option_test(str(Path(executable_dir) / executable), "", str(HERE))
            kernel_filter = input_row.get("kernel-name-filter", "")
            selected_range = (
                "@" + kernel_filter if kernel_filter else
                "0-" + str(args.kernel_number) if args.kernel_number > 0 else
                os.environ.get("DYNAMIC_KERNEL_RANGE", "")
            )
            preamble = (
                "set -e\nexport CUDA_VERSION=" + shlex.quote(cuda_version)
                + "\nexport CUDA_VISIBLE_DEVICES=" + shlex.quote(args.device_num)
                + "\nexport TRACES_FOLDER=" + shlex.quote(str(trace_root))
                + "\nexport DYNAMIC_KERNEL_RANGE=" + shlex.quote(selected_range)
                + "\nexport TERMINATE_UPON_LIMIT=" + ("1" if args.terminate_upon_limit else "0")
                + "\n"
            )
            mode = {"none": 0, "fast_forward": 1, "mark_region": 2}[args.spinlock_handling]
            command = shlex.quote(executable_path) + " " + str(input_row.get("args") or "")
            gpu_script = (
                preamble + "export TRACE_DEFER_FORMATTING=1\n"
                + "SPINLOCK_HANDLING_MODE=" + str(mode)
                + " SPINLOCK_ITER_TO_KEEP=" + str(args.spinlock_fast_forward_iterations)
                + " CUDA_INJECTION64_PATH=" + shlex.quote(str(tracer))
                + " " + command + "\n"
            )
            (directory / "run_gpu.sh").write_text(gpu_script)
            if mode:
                detector = HERE / "others/spinlock_tool/spinlock_tool.so"
                spin_script = preamble
                for phase in [0, 1]:
                    spin_script += (
                        "SPINLOCK_PHASE=" + str(phase) + " CUDA_INJECTION64_PATH="
                        + shlex.quote(str(detector)) + " " + command + "\n"
                    )
                (directory / "run_spinlock_detection.sh").write_text(spin_script)
                run_phase(directory, "run_spinlock_detection.sh", "gpu-spinlock")
            case = {"name": name, "directory": str(directory), "traces": str(trace_directory)}
            try:
                receipt = run_phase(directory, "run_gpu.sh", "gpu")
                # Kernel inventory names describe the final text files. The GPU
                # phase must have collected the corresponding raw record files.
                for inventory in trace_directory.glob("kernelslist*"):
                    if inventory.suffix == ".g":
                        continue
                    for line in inventory.read_text().splitlines():
                        if line.startswith("kernel-"):
                            base = line[:-3] if line.endswith(".xz") else line
                            if not (trace_directory / (base + ".records")).is_file():
                                raise RuntimeError("GPU record file is missing: " + base)
                case.update(status="collected", receipt=receipt)
            except Exception as error:
                case.update(status="failed", error=str(error))
            result["cases"].append(case)
            save(manifest, result)
            print("GPU", name, case["status"], flush=True)
    result["completed"] = True
    result["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save(manifest, result)
    return int(any(case["status"] != "collected" for case in result["cases"]))


def cpu_phase(args):
    root = args.results_dir.resolve()
    collected = json.loads((root / "gpu-traces.json").read_text())
    cases = [case for case in collected["cases"] if case["status"] == "collected"]
    result = {"phase": "cpu", "started_utc": datetime.now(timezone.utc).isoformat(), "cases": []}
    formatter = HERE / "tracer_tool/formatting/format-trace-capture"
    postprocessor = HERE / "tracer_tool/traces-processing/post-traces-processing"
    result["formatter_sha256"] = digest(formatter)
    result["postprocessor_sha256"] = digest(postprocessor)
    def process(case):
        directory = Path(case["directory"])
        trace_directory = Path(case["traces"])
        commands = [
            "set -e", "unset CUDA_INJECTION64_PATH LD_PRELOAD",
            "export CUDA_VISIBLE_DEVICES=",
        ]
        if any(trace_directory.glob("*.records")):
            commands.append(shlex.join([str(formatter), str(trace_directory)]))
        commands.append(
            shlex.join([str(postprocessor), str(trace_directory), "-j", str(args.postprocess_threads)])
            + (" --text" if args.postprocess_format == "text" else "")
        )
        (directory / "run_cpu.sh").write_text("\n".join(commands) + "\n")
        return run_phase(directory, "run_cpu.sh", "cpu")

    with ThreadPoolExecutor(max_workers=args.cpu_workers) as pool:
        futures = {pool.submit(process, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            outcome = {"name": case["name"]}
            try:
                outcome.update(status="processed", receipt=future.result())
            except Exception as error:
                outcome.update(status="failed", error=str(error))
            result["cases"].append(outcome)
            save(root / "cpu-traces.json", result)
            print("CPU", case["name"], outcome["status"], flush=True)
    result["completed"] = True
    result["ended_utc"] = datetime.now(timezone.utc).isoformat()
    save(root / "cpu-traces.json", result)
    return int(any(case["status"] != "processed" for case in result["cases"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["gpu", "cpu"])
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--traces-dir", type=Path)
    parser.add_argument("-B", "--benchmark_list", dest="benchmark_list", default="rodinia_2.0-ft")
    parser.add_argument("-D", "--device_num", dest="device_num", default="0")
    parser.add_argument("-l", "--limit_kernel_number", dest="kernel_number", type=int, default=-99)
    parser.add_argument("-t", "--terminate_upon_limit", action="store_true")
    parser.add_argument("--spinlock_handling", choices=["none", "fast_forward", "mark_region"], default="none")
    parser.add_argument("--spinlock_fast_forward_iterations", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--postprocess-threads", type=int, default=2)
    parser.add_argument("--postprocess-format", choices=["tracez", "text"], default="tracez")
    args = parser.parse_args()
    if args.phase == "cpu" and (args.cpu_workers < 1 or args.postprocess_threads < 1):
        parser.error("CPU worker and postprocess thread counts must be positive")
    return gpu_phase(args) if args.phase == "gpu" else cpu_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
