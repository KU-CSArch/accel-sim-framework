# Reproduce independent GPU capture and CPU processing

The [report](../../reports/trace-cpu-gpu-pipeline.md) defines the source boundary, complete measurements and coverage. Use this fork's `research/trace-cpu-gpu-pipeline` branch, based on Accel-Sim v2.0.0. [Effective inputs/tool identities](../../reports/data/trace-cpu-gpu-pipeline/two-phase/experiment.json) distinguish this experiment from the earlier postprocessing-only implementation.

## Run either phase

From the checkout root, with built tracer/formatter/postprocessor and an application suite defined under `util/job_launching/apps/`:

```bash
python3 util/tracer_nvbit/run_hw_trace.py --phase gpu \
  --results-dir "$RUN_ROOT" --traces-dir "$TRACE_ROOT" \
  -B "$SUITE" -D "$GPU_UUID"

python3 util/tracer_nvbit/run_hw_trace.py --phase cpu \
  --results-dir "$RUN_ROOT" --cpu-workers 2 --postprocess-threads 2
```

The first command does not start or wait for the second. To overlap CPU processing of a previous collection with the next GPU job, invoke the CPU phase independently and use a fresh results/trace namespace for the next GPU phase. CPU processing reads only the collected cases present when its command starts; it does not watch for future cases. The [tracer usage reference](../../../util/tracer_nvbit/README.md) describes controls and file ownership.

## Retained capsule and builds

The measured run is the `run` path in the linked experiment metadata; bulk trace bytes use its separate `traces` path. NCU reports, logs, model output and immutable inputs remain under `/mnt/hdd0/work/ncu/gpu-uarchsim/`; record/replay traces remain under `/mnt/hdd0/work/traces/`.

The capsule retains:

- Application binaries and source: `inputs/without_clock_instrumentation.cu`, per-image `fixture-defines.json`, `inputs/vector_add.cu`, `inputs/vector-add`, and all three instruction images.
- Real CUDA libraries under `inputs/native-runtime/`; simulator CUDA libraries under `inputs/runtime/`. Never substitute the simulator runtime into a native or NCU process.
- Executed deferred tracer `inputs/two-phase-tracer_tool.so`, CPU formatter, unchanged postprocessor, model binary and A6000 configuration. `inputs/tracer_tool.so` is the retained baseline input, not the deferred tracer used for the new measurements.
- Checksum-verified NVBit SDK copy `inputs/nvbit-sdk/`, [SDK archive identities](../../reports/data/trace-cpu-gpu-pipeline/two-phase/NVBIT-SDK-ARCHIVE.json), original SDK provenance and build receipts.
- CUDA 12.9 container recipe `inputs/Dockerfile.accel` and measured local image digest `sha256:6f6548f6b65f3a5313da9749da091ec0f55849e8a2b8813253cae61be0c52e60`. Rebuilding against mutable external packages can produce another image; record that new identity.

Install the retained SDK's `core/` and `LICENSE` into this checkout's `util/tracer_nvbit/nvbit_release/` before building. Do not link writable dependencies or binaries into another task's checkout. Build the tracer in the pinned image with this checkout mounted at `/work`:

```bash
make -C /work/util/tracer_nvbit/tracer_tool -j4 ARCH=sm_86
make -C /work/util/tracer_nvbit/tracer_tool/traces-processing
```

The formatter also builds with ordinary C++ without CUDA:

```bash
make -C util/tracer_nvbit/tracer_tool/formatting
ldd util/tracer_nvbit/tracer_tool/formatting/format-trace-capture
```

The three instruction images use KIND/CHAINS/SEPARATED definitions from the retained fixture manifest and the [original source](https://github.com/KU-CSArch/gpu-uarchsim/blob/bbe7d333e55d7b3f36250b254f764f5718b3f529/AE/benchmarks/without_clock_instrumentation.cu). Build with CUDA 12.9:

```text
nvcc -O3 -std=c++17 --cudart shared --no-compress
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_86,code=compute_86 -Xptxas=-v
  -DKIND=<kind> -DCHAINS=1 -DSEPARATED=0
  without_clock_instrumentation.cu -o <image>
```

The [vector fixture](two-phase/vector_add.cu) builds with:

```text
nvcc -O3 -std=c++17 --cudart shared --no-compress -lineinfo
  -gencode arch=compute_86,code=sm_86 vector_add.cu -o vector-add
```

## Fresh repeated hardware experiments

Create a fresh task-owned run directory (`NEW_RUN`) and trace directory on the selected HDD namespaces. Copy the immutable retained `inputs/` into `$NEW_RUN/inputs/`. Copy the exact [two-phase drivers](two-phase/) into `$NEW_RUN/`. The source hashes in that directory identify the executed bodies. Copy the linked experiment metadata to `$NEW_RUN/experiment.json` and update `run`, `traces`, `worktree` and `gpu` for the new run. Its `cases` list is the three instruction image names. Record hashes for any rebuilt tool or application.

Create three local benchmark definitions from the retained YAML files, updating their `exec_dir` and `data_dirs` to the new inputs:

| File under `util/job_launching/apps/` | Retained definition | Suite |
|---|---|---|
| `define-trace-pipeline-validation.yml` | `inputs/benchmark-definition.yml` | `trace-pipeline-validation` |
| `define-trace-phase-vector-add.yml` | `inputs/vector-benchmark.yml` | `trace-phase-vector-add` |
| `define-trace-phase-replay.yml` | `inputs/replay-benchmark.yml` | `trace-phase-replay-imad` |

Use an otherwise idle target GPU. The drivers do not stop or pause any other collector:

```bash
python3 "$NEW_RUN/validate_two_phase_batches.py"
python3 "$NEW_RUN/validate_vector_batches.py"
python3 "$NEW_RUN/validate_phase_independence_attempt02.py"
```

The first two scripts rotate inline versus separate phases across five repetitions. Every capture runs the same full input four times, the first three warmups. All numerical outputs are checked. Existing output directories cause an error; retain them and select fresh namespaces instead of overwriting observations. These scripts execute at module scope; do not import them as helper modules.

The hold test captures FADD, starts the actual separate CPU formatter, identifies and pauses only that descendant, runs six native processes and eight NCU collections, then captures IMAD before releasing the formatter. A `finally` block releases that process and waits for its CPU phase. This is a deliberately paused causal experiment, not a timing benchmark. NCU uses explicit real CUDA libraries, no injected tracer and `--clock-control none --cache-control none`. Import each saved `.ncu-rep` using `ncu --import <report> --page raw --csv --log-file <metrics.csv>` before the figure reduction; all eight must contain the requested metrics.

Run workflow checks independently:

```bash
python3 -m unittest discover -s util/tracer_nvbit/tests -v
```

## Original-writer and grouped-record comparisons

The [reference writer](two-phase/reference_writer.cpp) retains the original v2.0.0 receiver's formatting body. The shared reader loads the saved record ABI/metadata, but instruction formatting is independent of `TraceFormatter`. TMA API stubs fail explicitly; this reference is limited to the stated A6000 inputs.

In the pinned image, mount this checkout at `/work` and the new run at `/results`. After building the CPU objects, compile:

```bash
g++ -O2 -std=c++17 -I/work/util/tracer_nvbit/nvbit_release/core \
  -I/usr/local/cuda/include /results/reference_writer.cpp \
  /work/util/tracer_nvbit/tracer_tool/formatting/trace_capture.o \
  /work/util/tracer_nvbit/tracer_tool/formatting/trace_formatter.o \
  -o /results/reference-writer
```

Compile the existing one-CTA [verifier](verify_tracez.cpp) with the checkout mounted at `/source` and zstd development headers/libraries:

```bash
g++ -O2 -std=c++17 /results/verify_tracez.cpp -lzstd -o /results/verify_tracez
```

Compile [the all-CTA/warp verifier](two-phase/verify_tracez_all_warps.cpp) the same way:

```bash
g++ -O2 -std=c++17 /results/verify_tracez_all_warps.cpp \
  -lzstd -o /results/verify_tracez_all_warps
```

Run the record checkers only after their capture/CPU phases finish:

```bash
python3 "$NEW_RUN/verify_two_phase_records.py" "$NEW_RUN"
python3 "$NEW_RUN/verify_vector_records.py" "$NEW_RUN"
```

The instruction checker includes four smoke kernels in its 72-kernel contract. Reproduce that smoke capture first with `--phase gpu --results-dir "$NEW_RUN/smoke" --traces-dir "$NEW_TRACES/smoke" -B trace-pipeline-validation:fadd-c1-o0 -D "$GPU_UUID"`, then `--phase cpu --results-dir "$NEW_RUN/smoke" --postprocess-format text`. The vector checker covers every CTA/warp of all 20 vector kernels. Both compare full text bytes from the original writer, not only counts or checksums. Their population assertions define these fixed validation experiments; they are not production trace restrictions.

The [formatting matrix](two-phase/formatting_matrix.cpp) builds with the same CPU objects. Run `formatting-matrix <smoke-kernel.trace.records> <fresh-matrix-directory>`; for each of its 48 record files run `reference-writer <record-file> <reference-text>`, then compare the generated `.trace` against that reference byte for byte. It tests 16 output-control combinations × three existing spinlock modes using the first/last 1,024 actual records. It does not claim real spinlock or BAR/SYNCS value-path validation.

## Cycle-model input preservation

Collect the regression-only IMAD suite into `$NEW_RUN/model-capture` and `$NEW_TRACES/model-capture` with the two explicit launcher commands, selecting `-B trace-phase-replay-imad` for GPU capture. That fixture uses the same IMAD binary with `1 128 1 1 1 3`; the smaller input tests encoding preservation, not GPU performance.

```bash
python3 "$NEW_RUN/validate_model_equivalence.py" "$NEW_RUN"
```

This creates grouped text with the original writer and compares four independent unchanged A6000 cycle-model runs per format against the deferred `.tracez` from those same records. It uses explicit simulator CUDA libraries, no GPU devices, copied configuration and the pinned image. It requires identical per-kernel cumulative cycles and instruction counts. All commands and logs are saved. This does not validate simulator timing against hardware.

## Render from saved results

```bash
python3 "$NEW_RUN/analyze_two_phase.py" "$NEW_RUN"
```

The reduction reads both batch JSON files, the successful hold-test events and the eight imported NCU metric CSVs. It writes all 20 batch values, min/median/max summaries, every individual timing point and the process-independence timeline. Intentional CPU pause time is never used in throughput comparisons.

The previous `validate_gpu_batches.py`, `verify_gpu_batches.py`, `analyze_gpu_batches.py` and their original datasets remain historical evidence for source `3576fff9018057a6c5e3ccd990fba91c7ae2c497`. They use the removed bounded postprocessing-queue interface. Reproduce them from that exact source revision in a separate checkout; they do not implement the current two-phase boundary.
