#!/usr/bin/env python3

from optparse import OptionParser
import os
import subprocess
import shlex
from trace_pipeline import TracePipeline
import os

this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
import sys

sys.path.insert(0, os.path.join(this_directory, "..", "job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml
import common
import re
import datetime

# We will look for the benchmarks
parser = OptionParser()
parser.add_option(
    "-B",
    "--benchmark_list",
    dest="benchmark_list",
    help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for "
    + "the benchmark suite names.",
    default="rodinia_2.0-ft",
)
parser.add_option(
    "-D", "--device_num", dest="device_num", help="CUDA device number", default="0"
)
parser.add_option(
    "-n",
    "--norun",
    dest="norun",
    action="store_true",
    help="Do not actually run the apps, just create the dir structure and launch files",
)
parser.add_option(
    "-l",
    "--limit_kernel_number",
    dest="kernel_number",
    default=-99,
    type="int",
    help="Sets a hard limit to the number of traced kernels",
)
parser.add_option(
    "-t",
    "--terminate_upon_limit",
    dest="terminate_upon_limit",
    action="store_true",
    help="Once the kernel limit is reached, terminate the tracing process",
)
parser.add_option(
    "--spinlock_handling",
    dest="spinlock_handling",
    choices=["none", "fast_forward", "mark_region"],
    default="mark_region",
    help="How to handle spinlock instructions: none, fast_forward, or mark_region (default: mark_region)",
)
parser.add_option(
    "--spinlock_fast_forward_iterations",
    dest="spinlock_fast_forward_iterations",
    type=int,
    default=1,
    help="Number of iterations to keep for spinlock fast forwarding. Only used if spinlock_handling is fast_forward",
)

parser.add_option(
    "--postprocess-workers", type="int", default=0,
    help="Overlap capture with this many CPU postprocess workers; 0 runs serially",
)
parser.add_option(
    "--max-pending-postprocess", type="int", default=None,
    help="Maximum completed bundles being processed or queued (default: twice workers)",
)
parser.add_option(
    "--results-dir", default=None,
    help="Trace output root instead of hw_run/traces/device-N/CUDA_VERSION",
)
parser.add_option(
    "--keep-raw", action="store_true", default=False,
    help="Retain raw traces after successful postprocessing",
)
parser.add_option(
    "--postprocess-format", type="choice", choices=["tracez", "text"],
    default="tracez", help="Replay trace format (default: tracez)",
)
parser.add_option(
    "--postprocess-threads", type="int", default=8,
    help="Kernel threads per CPU postprocess worker (default: 8)",
)

(options, args) = parser.parse_args()
if options.postprocess_workers < 0:
    parser.error("--postprocess-workers must be nonnegative")
if options.max_pending_postprocess is not None and options.max_pending_postprocess < 1:
    parser.error("--max-pending-postprocess must be positive")
if options.postprocess_threads < 1:
    parser.error("--postprocess-threads must be positive")


common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version(this_directory)
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv"

nvbit_tracer_path = os.path.join(this_directory, "tracer_tool")
nvbit_spinlock_path = os.path.join(this_directory, "others", "spinlock_tool")

with TracePipeline(options.postprocess_workers, options.max_pending_postprocess) as pipeline:
    for bench in benchmarks:
        edir, ddir, exe, argslist = bench
        for argpair in argslist:
            args = argpair["args"]
            kernel_name_filter = argpair.get("kernel-name-filter", "") if isinstance(argpair, dict) else ""
            run_name = os.path.join(exe, common.get_argfoldername(argpair))
            this_run_dir = os.path.abspath(
                os.path.expandvars(
                    os.path.join(
                        this_directory,
                        "..",
                        "..",
                        "hw_run",
                        "traces",
                        "device-" + options.device_num,
                        cuda_version,
                        run_name,
                    )
                )
            )
            if options.results_dir:
                this_run_dir = os.path.abspath(os.path.join(
                    os.path.expandvars(options.results_dir), run_name
                ))
            pipeline.wait_directory(this_run_dir)
            this_trace_folder = os.path.join(this_run_dir, "traces")
            if not os.path.exists(this_run_dir):
                os.makedirs(this_run_dir)
            if not os.path.exists(this_trace_folder):
                os.makedirs(this_trace_folder)

            # link the data directory
            try:
                benchmark_data_dir = common.dir_option_test(
                    os.path.join(ddir, exe, "data"), "", this_directory
                )
                if os.path.lexists(os.path.join(this_run_dir, "data")):
                    os.remove(os.path.join(this_run_dir, "data"))
                os.symlink(benchmark_data_dir, os.path.join(this_run_dir, "data"))
            except common.PathMissing:
                pass

            all_data_link = os.path.join(this_run_dir, "data_dirs")
            if os.path.lexists(all_data_link):
                os.remove(all_data_link)
            top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
            os.symlink(top_data_dir_path, all_data_link)

            if args == None:
                args = ""
            exec_path = common.file_option_test(os.path.join(edir, exe), "", this_directory)
            sh_contents = "set -e\n"

            if options.terminate_upon_limit:
                sh_contents += "export TERMINATE_UPON_LIMIT=1; "

            if "mlperf" in exec_path:
                # For mlperf by default we turn this flag on
                sh_contents += "export TERMINATE_UPON_LIMIT=0; "
                exec_path = ". " + exec_path

                if options.kernel_number > 0:
                    sh_contents +=  ('\nexport DYNAMIC_KERNEL_RANGE="0-'+str(options.kernel_number)+'"\n')
                else:
                    sh_contents +=  ('\nexport DYNAMIC_KERNEL_RANGE="0-'+str(50)+'"\n')
            else:
                if kernel_name_filter:
                    sh_contents +=  (f'\nexport DYNAMIC_KERNEL_RANGE="@{kernel_name_filter}"\n')
                elif options.kernel_number > 0:
                    sh_contents +=  ('\nexport DYNAMIC_KERNEL_RANGE="0-'+str(options.kernel_number)+'"\n')
                else:
                    sh_contents +=  (f'\nexport DYNAMIC_KERNEL_RANGE="{os.environ.get("DYNAMIC_KERNEL_RANGE", "")}"\n')

            # Capture owns the GPU. Grouping and output compression need only CPU.
            sh_contents += (
                '\nexport CUDA_VERSION="'
                + cuda_version
                + '"; export CUDA_VISIBLE_DEVICES="'
                + options.device_num
                + '" ; '
            )
        
            # Map spinlock handling mode to integer
            spinlock_mode_map = {"none": 0, "fast_forward": 1, "mark_region": 2}

            capture_contents = (
                sh_contents
                + "\nrm -f traces/*"
                + "\nexport TRACES_FOLDER=" + shlex.quote(this_run_dir)
                + f"; SPINLOCK_HANDLING_MODE={spinlock_mode_map[options.spinlock_handling]} SPINLOCK_ITER_TO_KEEP={options.spinlock_fast_forward_iterations} CUDA_INJECTION64_PATH="
                + shlex.quote(os.path.join(nvbit_tracer_path, "tracer_tool.so"))
                + " " + exec_path + " " + str(args) + "\n"
            )
            postprocess_contents = (
                "set -e\n"
                "unset CUDA_INJECTION64_PATH LD_PRELOAD\n"
                "export CUDA_VISIBLE_DEVICES=\n"
                + shlex.quote(os.path.join(
                    nvbit_tracer_path, "traces-processing", "post-traces-processing"
                )) + " " + shlex.quote(this_trace_folder)
                + " -j " + str(options.postprocess_threads)
                + (" --text" if options.postprocess_format == "text" else "") + "\n"
            )
            if not options.keep_raw:
                postprocess_contents += (
                    "rm -f " + shlex.quote(this_trace_folder) + "/*.trace "
                    + shlex.quote(this_trace_folder) + "/*.trace.xz "
                    + shlex.quote(this_trace_folder) + "/kernelslist\n"
                )
            tracer_contents = "set -e\nbash run_capture.sh\nbash run_postprocess.sh\n"

            # Spinlock tool run script
            # will run twice, once for phase 0 and once for phase 1
            spinlock_contents = (
                sh_contents
                + "\nrm -f spinlock_detection/*"
                + "\nexport TRACES_FOLDER="
                + this_run_dir
                + "; SPINLOCK_PHASE=0 CUDA_INJECTION64_PATH="
                + os.path.join(nvbit_spinlock_path, "spinlock_tool.so")
                + " "
                + exec_path
                + " "
                + str(args)
                + " ; "
                + " SPINLOCK_PHASE=1 CUDA_INJECTION64_PATH="
                + os.path.join(nvbit_spinlock_path, "spinlock_tool.so")
                + " "
                + exec_path
                + " "
                + str(args)
                + " ; "
            )

            for path, content in [("run.sh", tracer_contents), ("run_capture.sh", capture_contents), ("run_postprocess.sh", postprocess_contents), ("run_spinlock_detection.sh", spinlock_contents)]:
                with open(os.path.join(this_run_dir, path), "w") as script:
                    script.write(content)
                if subprocess.call(["chmod", "u+x", os.path.join(this_run_dir, path)]) != 0:
                    exit(f"Error chmod {path} runfile")


            if not options.norun:
                print("Running {0}".format(exe), flush=True)
                pipeline.capture(
                    this_run_dir,
                    spinlock=options.spinlock_handling in ["fast_forward", "mark_region"],
                )
