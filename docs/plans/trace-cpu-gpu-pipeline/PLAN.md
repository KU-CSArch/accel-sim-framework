# Independent GPU capture and CPU trace processing

The accepted direction is two explicit script phases: GPU capture saves its records and metadata and returns; CPU trace processing reads that collection separately. CPU formatting, serialization, grouping, compression and worker completion must never be GPU-phase dependencies. No GPU phase waits for a CPU queue, CPU result or CPU process. Each phase has its own output status file.

Checked checkout: /home/inje/workspace/gpu/gpu-uarchsim/worktree/accel-sim-trace-cpu-gpu
Checked head: 3576fff9018057a6c5e3ccd990fba91c7ae2c497
Framework base: Accel-Sim v2.0.0 64653015f85fb5664c84a10f48527e8897d289d0

The retained proposed/ bodies and PLAN-IDENTITIES.json describe the postprocessing-only implementation at 3576fff. They are historical evidence, not the active plan. The two-phase-source/ bodies and TWO-PHASE-IDENTITIES.json define the current implementation.

## Current complete flow and hold

instrument_function_if_needed injects instrument_inst before selected SASS instructions. The injected function supplies inst_trace_t to ChannelDev::push. The channel is 1 MiB. ChannelDev::flush waits for the host doorbell to be cleared; ChannelHost::recv copies one completed buffer and releases the doorbell. recv_thread_fun then spends CPU time formatting every record before it calls recv again. That delays release of the next full channel. leave_kernel_launch synchronizes, emits a flush marker and waits for receiver completion, then closes the raw xz stream. The previous launcher adds CPU queue-depth waits and joins all CPU workers. Both kinds of CPU dependency violate the requested phase boundary.

## Intended owners and values

GPU phase retains the injected function, record ABI, channel size, counters, collection policy, numerical program and CUDA synchronization. Its receiver bulk-writes the original inst_trace_t records for each received buffer to a .trace.records file. It does not format instruction text, run address compression, group records, compress files or launch a CPU worker. The static kernel header and formatting options are saved along with the full opcode-ID mapping and spinlock metadata. The terminal channel marker is not an instruction record. Footer completion and file close occur before the capture process returns and the GPU case is published as collected.

CPU phase reads the saved record blocks, original kernel header, options, opcode mapping and spinlock metadata. TraceFormatter owns the moved per-warp spinlock/replay state and the original text writer. The CPU phase emits the same raw text/xz format, then invokes the unchanged grouping/compression tool. Only the CPU phase creates or joins CPU workers. It makes no CUDA call, does not import application discovery or call nvcc, and clears CUDA visibility/injection/preload variables for its process.

The SDK's TMA parsing/address APIs explicitly take a live CUDA context. For TMA instructions only, GPU acquisition saves the values returned by those existing APIs, including inbound addresses, while that context exists. The CPU phase consumes the saved values; it does not reopen a CUDA context or invent a TMA decoder. This context-bound acquisition remains live work. The RTX A6000 target does not execute TMA, so hardware validation of this TMA path is target-missing. It is not described as hardware-validated.

The native 1,296-byte record is persisted without per-field packing, so all lane values, addresses, immediates and union bytes remain available. The file records its native byte-order marker, record size, alignment and union offset; a CPU reader must satisfy that recorded ABI. The intermediate storage is large: the inspected long synchronization input requires 97,845,905,664 record bytes before headers. No new GPU buffer, speculative data reconstruction, instruction filtering or compensating execution logic is added. Required channel draining and file I/O can still exert physical backpressure; CPU formatting and CPU queue completion cannot.

The two added host maps are ctx_formatters (one moved formatter state per current context) and ctx_deferred_files (open record-file handles that must receive metadata/footer before context teardown). They hold host file/formatting state, not GPU model resources. The formatter retains the same per-warp spinlock counters/replay state that previously lived in recv_thread_fun.

## Exact source and caller replacements

The complete files in two-phase-source/ are prepared before source mutation:

- trace_record.h -> util/tracer_nvbit/tracer_tool/trace_record.h: the unchanged POD record/type declarations, separated from managed CUDA counters.
- common.h -> util/tracer_nvbit/tracer_tool/common.h: includes the POD declarations and preserves the original managed counters.
- trace_formatter.h/.cpp -> util/tracer_nvbit/tracer_tool/formatting/: complete CPU-only formatter, saved option/TMA data and moved per-warp state. Instruction formatting is the original receiver body; the TMA API results are passed as captured data.
- trace_capture.h/.cpp -> the same formatting directory: native record block persistence, explicit metadata/footer and CPU record reader. Record bytes are copied in bulk, not reinterpreted during collection.
- format_trace_capture.cpp and formatting.Makefile -> formatting/format_trace_capture.cpp and formatting/Makefile: CPU-only command and C++ build.
- tracer_tool.cu and tracer.Makefile -> tracer_tool.cu and Makefile: exact producer/receiver/lifecycle and build replacements. The .cu device instrumentation body inject_funcs.cu is unchanged. Traced graph files left open by existing callbacks receive their footer at context termination; this does not repair the existing graph attribution model.
- run_hw_trace.py -> util/tracer_nvbit/run_hw_trace.py: required --phase gpu|cpu; GPU phase owns gpu-traces.json, CPU phase owns cpu-traces.json. GPU does not generate or run CPU scripts. CPU worker count is scoped to CPU phase. --results-dir identifies the collection; --traces-dir places trace bytes separately. Fresh GPU runs have distinct namespaces and never overwrite a collection that a CPU phase might consume. Each per-case trace directory is created with exist_ok=False; a separate results directory cannot reuse an existing trace directory. The corresponding test retains the first owner's record bytes across that rejected second capture.
- trace_pipeline.py -> util/tracer_nvbit/trace_pipeline.py: retains only the command/receipt function. The CPU-coupled TracePipeline class and its admission/join path are removed.
- test_trace_pipeline.py -> util/tracer_nvbit/tests/test_trace_pipeline.py: complete tests in the candidate source set cover the explicit phase contract, CPU execution without CUDA discovery and GPU return while a CPU job is held.
- util/tracer_nvbit/README.md, docs/plans/trace-cpu-gpu-pipeline/, docs/reports/trace-cpu-gpu-pipeline.md and reproduction scripts: reflect the two phases, their exact scope and measured evidence. Preserve the previous run, source revision and original identities as historical evidence; do not relabel its 4.30% postprocessing result as full detachment.

No production source file is changed before these bodies and replacements are complete. The build first removes only this checkout's tracer_tool.so symlink; its retained baseline target is never overwritten. Copy the pinned NVBit SDK as an owned dependency input into this checkout and record its hashes. Build and run only in this task checkout and namespace. The unchanged postprocessor may be read from this workspace's retained baseline input, never another workspace's writable build output.

## Required validation and independent boundaries

1. Build the CPU formatter with ordinary C++ and show it has no CUDA/NVBit linkage. Build the tracer in the pinned CUDA 12.9 image. Verify the record ABI and unchanged injected source.
2. Test raw-record round trips and complete formatting against the original v2.0.0 writer using the same captured record data, including spinlock/replay, address formatting, core/line fields and lane-value controls. Use ordinary file-format errors, not production timing gates or test-specific execution paths.
3. Conduct real RTX A6000 captures with matching application binaries/inputs, three full-work warmups and one measured-work launch. Repeat independent batches at least five times per timing setting and report all min/median/max values. Verify numerical outputs and per-kernel instruction populations. Native performance never uses instrumented capture timings.
4. Hold the separate CPU formatter after a GPU capture. Run the next real native application, NCU collection and another GPU capture to completion before releasing that CPU process. Preserve process identities and host event timestamps. This is a causal independence experiment, not a throughput measurement; exclude its intentional hold from speed claims.
5. Compare every serialized record and grouped instruction sequence, including masks, addresses and values. Run the unchanged A6000 cycle model on equivalent text/tracez inputs in four independent processes and compare per-kernel cycles.
6. Use only this workspace's new run/trace paths. Hold the existing GPU0 caller only after its currently running capture can finish; user permission for that boundary persists. Do not stop a running GPU child, alter its source or move its outputs. Resume or explicitly preserve the stopped caller state when the bounded validation ends.
