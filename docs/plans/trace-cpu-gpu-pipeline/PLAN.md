# CPU/GPU trace pipeline

Task checkout: /home/inje/workspace/gpu/gpu-uarchsim/worktree/accel-sim-trace-cpu-gpu
Branch: research/trace-cpu-gpu-pipeline
Baseline: Accel-Sim v2.0.0, 64653015f85fb5664c84a10f48527e8897d289d0
Fork: https://github.com/KU-CSArch/accel-sim-framework

The user authorized separating CPU and GPU tracing work and creating this fork/branch. This changes collection orchestration, not GPU-uArchSim architecture or Accel-Sim timing behavior. Existing experiments keep their frozen collectors and binaries.

## Source audit and current chain

Read run_hw_trace.py, its common.py benchmark/argument/path/version producers, tracer_tool.cu kernel-end flush and recv thread, inject_funcs.cu channel producer and post-traces-processing.cpp main/group_per_block. The injected function produces one warp record through ChannelDev::push. recv_thread_fun drains that channel concurrently, formats records and writes/compresses the raw trace. Kernel exit synchronizes the device, flushes the channel and waits for the receiver before closing the raw file. The launcher then waits for the separate post-traces-processing executable before it can launch another GPU capture. The postprocessor groups the completed per-kernel records by CTA and warp and writes the replay trace and final kernel list. It has no CUDA runtime dependency.

The source study distinguishes live trace production and channel draining from processing a completed bundle. This change retains the Accel-Sim text/binary replay formats and introduces no dependency on another simulator.

## Changed owners and complete source

- util/tracer_nvbit/run_hw_trace.py: existing benchmark loop and generated shell scripts. The complete replacement is proposed/run_hw_trace.py.
- util/tracer_nvbit/trace_pipeline.py: host orchestration only. Complete declarations and method bodies are proposed/trace_pipeline.py. run_phase owns one subprocess and its logs/receipt. TracePipeline owns the bounded pending CPU futures; the main caller owns the single GPU capture stream. CPU tasks have CUDA_VISIBLE_DEVICES empty and CUDA injection/preload removed. No worker calls chdir.
- util/tracer_nvbit/tests/test_trace_pipeline.py: process-level tests of ordering, bounded overlap and failure behavior.
- util/tracer_nvbit/README.md: options, exact phase boundary, retained evidence and source references.
- docs/plans/trace-cpu-gpu-pipeline/: portable copies of this plan and both complete proposed source files.
- docs/reports/trace-cpu-gpu-pipeline.md: measured validation and remaining scope.

The complete proposed source files are prepared before production files are changed. Their source hashes are recorded in PLAN-IDENTITIES.json. No methods are omitted or left for implementation-time choice.

## Intended progression

For each benchmark/input, the main thread generates capture and postprocess scripts, waits for a previous task using the same directory, then runs any existing spinlock prepass and the capture script. Raw trace files close before submission to a CPU worker. The main caller can immediately launch the next capture while the worker groups and compresses that previous bundle. A bounded count of completed/in-flight CPU bundles limits queued data; it can deliberately stop admission if the CPU falls behind. The existing raw-file cleanup runs only after successful postprocessing unless --keep-raw is selected. Failure receipts are retained, errors produce a nonzero launcher exit, and context exit waits for submitted workers. Repeated directories wait for their prior CPU task before any generated script or trace is overwritten. With zero workers, each CPU phase completes before the next capture.

Generated run.sh still executes both stages serially for manual use. Existing kernel filters, trace contents, warmups, spinlock semantics and postprocessor format selection stay with their original owners. --postprocess-threads controls the existing -j option; --postprocess-format selects the existing text or tracez output (default tracez); no model latency/configuration changes are introduced. Kernel-limit input is explicitly parsed as an integer because the existing owner compares it numerically.

## Discriminating validation

1. Run process-level tests: CPU1 must overlap capture2; the CPU task must have no visible CUDA device or injection environment; queue-depth one must force CPU1 completion before capture2; capture failure must not schedule CPU work; CPU failure must propagate; repeated directories must wait.
2. Generate scripts through the real launcher with a controlled benchmark definition and verify the default serial script and --norun behavior.
3. Reprocess retained real raw traces in serial and pipelined layouts. Compare grouped instruction sequences, masks, addresses and launch order after normalizing run-specific header paths only. Run the unchanged pinned Accel-Sim binary/configuration on both outputs and compare actual kernel cycle counts.
4. Run matched real GPU collection batches in serial and overlapping modes when GPU0's current capture has drained, without concurrent NCU/capture on that GPU. Preserve exact inputs, binaries, commands, source/configuration hashes, five independent batches in each mode, per-phase host timestamps and numerical output checks. Report median and min-max host elapsed times and actual overlap; do not equate capture-process duration with GPU busy time or make a speedup claim from workflow tests.

No existing GPU measurements are interrupted or moved. The idle parent that queues the next GPU0 batch may be held and restored only at its existing natural waiting boundary, with PID/start-time verification and restoration on every exit. Running GPU work and model children are left alone. New output uses this workspace's namespace and /mnt/hdd0/work/traces/ for traces.

## Claim boundary

This first change removes the post-capture CPU dependency from next-input GPU admission. It does not detach GPU instrumentation, channel draining or raw text formatting/compression within recv_thread_fun. Those remain part of live capture. A fully deferred raw-record serializer needs its own persisted record/schema design and equivalence validation; no such unvalidated format is introduced here.

The generated-script writer closes files explicitly before launching them. Preserve the launcher executable mode (0755).

Each receipt and subprocess failure names the absolute generated phase script, so asynchronous failures retain their case directory in the reported command.

The GPU experiment additionally tests the existing TRACE_FILE_COMPRESS=0 setting with overlapping workers and the existing tracez output format. This removes raw xz compression from live capture; final grouped tracez compression stays in the CPU postprocessor. Raw text formatting and channel draining remain online. Five independent batches per condition rotate three conditions to expose contention or regression rather than assuming speedup.
