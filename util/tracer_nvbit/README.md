# Accel-Sim GPU trace capture and CPU processing

The launcher has two explicit phases. The GPU phase saves dynamic instruction records and returns. The CPU phase reads that saved collection and produces replay traces. Start the next native run, NCU collection or GPU capture after the GPU command returns; CPU processing is a separate command.

Build the NVBit tracer and the CPU tools in this checkout before running. `tracer_tool/Makefile` also builds `formatting/format-trace-capture`. The postprocessor requires zstd development files. The standalone formatter can be built without CUDA:

```bash
make -C tracer_tool/formatting
```

## GPU phase

Select a fresh results directory and a fresh trace namespace. Benchmark suites and application inputs are defined under [job_launching/apps](../job_launching/apps).

```bash
python3 run_hw_trace.py --phase gpu \
  --results-dir "$RUN_ROOT" --traces-dir "$TRACE_ROOT" \
  -B rodinia_2.0-ft -D 0
```

The generated `run_gpu.sh` enables `TRACE_DEFER_FORMATTING=1`. NVBit instrumentation and its channel remain live. The receiver saves full record blocks; it does not format instruction text, compress addresses, run xz, group warps or wait for CPU workers. Kernel completion still drains and closes its record file before the case is published as collected in `gpu-traces.json`.

Trace bytes are written under `$TRACE_ROOT/<application>/<input>/traces/`. Logs, numerical application outputs, scripts and receipts are under `$RUN_ROOT/<application>/<input>/`. Omitting `--traces-dir` puts trace bytes under `$RUN_ROOT/collected-traces/`. Existing per-case trace directories and an existing GPU manifest are preserved; use new namespaces for a new collection.

Required channel draining and record-file I/O remain on the live path. A slow disk can still backpressure capture. Context-bound NVBit TMA address acquisition also remains live; its formatting is deferred. The RTX A6000 validation does not exercise TMA.

## CPU phase

```bash
python3 run_hw_trace.py --phase cpu --results-dir "$RUN_ROOT" \
  --cpu-workers 2 --postprocess-threads 2
```

This command reads the cases marked collected in a snapshot of `gpu-traces.json`. It does not discover applications or invoke `nvcc`. Each CPU worker runs `run_cpu.sh`: format saved records, then invoke the existing postprocessor. CUDA visibility, injection and preload are cleared for these processes. `cpu-traces.json` records their independent outcomes. CPU failure leaves the GPU manifest and saved records intact. There is no CPU queue or final CPU join in the GPU command.

The default replay format is `.tracez`; `--postprocess-format text` selects `.traceg`. The CPU phase also accepts previously collected raw `.trace`/`.trace.xz` inputs recorded in the same manifest layout. Raw and intermediate files are retained. CPU workers and each postprocessor's `-j` threads apply only to this phase. Run one CPU phase per collection at a time.

Separate collections can progress independently:

```bash
python3 run_hw_trace.py --phase cpu --results-dir "$PREVIOUS_RUN" &
python3 run_hw_trace.py --phase gpu --results-dir "$NEXT_RUN" \
  --traces-dir "$NEXT_TRACES" -B rodinia_2.0-ft -D 0
```

The second command has no dependency on completion of the first. This does not promise exclusive CPU or storage bandwidth.

## Files and retained controls

| File | Produced by | Purpose |
|---|---|---|
| `kernel-*.trace.records` | GPU phase | Native record blocks, original kernel header, formatting options, opcode names and required metadata |
| `kernelslist_ctx_*`, `stats_ctx_*` | GPU phase | Existing kernel/memcpy inventory and instruction counters |
| `kernel-*.trace` or `.trace.xz` | CPU formatter | Original textual instruction format |
| `kernel-*.tracez` or `.traceg` | CPU postprocessor | Records grouped by CTA and warp for replay |
| `kernelslist.g` | CPU postprocessor | Replay inventory |

Saved records preserve the native `inst_trace_t` layout. The file records its byte order, size, alignment and union offset; the reader requires that ABI. With the validated SDK/build, a record occupies 1,296 bytes. These intermediate files can be much larger than text or final compressed replay traces. Keep collection/replay tools with the source, SDK and build identities used to create the files.

`TRACE_FILE_COMPRESS` chooses the CPU formatter's raw-text compression in deferred mode; it does not compress the GPU record file. Direct tracer use retains the original inline writer by default (`TRACE_DEFER_FORMATTING=0`). Set `TRACE_DEFER_FORMATTING=1` explicitly for direct record capture, then run `tracer_tool/formatting/format-trace-capture <records-file-or-directory>` separately.

The launcher retains kernel selection (`-l`, `-t`, input `kernel-name-filter`, or `DYNAMIC_KERNEL_RANGE`), optional source lines (`TRACE_LINEINFO` with application `-lineinfo`), and explicit spinlock options. Spinlock detection, when selected, still uses the existing detection prepass; it is not covered by the hardware detachment measurements. No filtering, device instrumentation or cycle-model change is introduced by phase separation.

See the [source plan](../../docs/plans/trace-cpu-gpu-pipeline/PLAN.md), [measured validation](../../docs/reports/trace-cpu-gpu-pipeline.md) and [reproduction commands](../../docs/experiments/trace-cpu-gpu-pipeline/README.md).
