# Reproduce the CPU/GPU trace pipeline experiment

Use the source branch `research/trace-cpu-gpu-pipeline`, based on Accel-Sim v2.0.0. The [report](../../reports/trace-cpu-gpu-pipeline.md) defines the measurements and their limits.

## Workflow tests

```bash
python3 -m unittest discover -s util/tracer_nvbit/tests -v
```

## Hardware run

The retained capsule is the directory recorded as `run` in [experiment.json](../../reports/data/trace-cpu-gpu-pipeline/experiment.json). It contains immutable application binaries, real CUDA and simulated CUDA runtimes, model configurations, the unchanged NVBit tracer, the postprocessor, benchmark source and per-image compile definitions. Verify them against [input-sha256.json](../../reports/data/trace-cpu-gpu-pipeline/input-sha256.json). The [benchmark source](https://github.com/KU-CSArch/gpu-uarchsim/blob/bbe7d333e55d7b3f36250b254f764f5718b3f529/AE/benchmarks/without_clock_instrumentation.cu) has SHA-256 `c9e033c18b372a46a468bb28c9ea608ba26951eadc342555e1f3e3030daae09d`. The three images use KIND/CHAINS/SEPARATED definitions recorded in [fixture-defines.json](../../reports/data/trace-cpu-gpu-pipeline/fixture-defines.json) and this compile command in the pinned CUDA 12.9 image:

```text
nvcc -O3 -std=c++17 --cudart shared --no-compress
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_86,code=compute_86 -Xptxas=-v
  -DKIND=<kind> -DCHAINS=1 -DSEPARATED=0
  without_clock_instrumentation.cu -o <image>
```

Create a fresh task-owned run directory and trace directory. Copy the retained `inputs/`, the supplied `validate_gpu_batches.py`, and `experiment.json` into the new run directory. Update the manifest's `worktree`, `run`, `traces` and `gpu` paths/identity for the selected checkout and target device. Rebuild the tracer/postprocessor in that checkout or install the hash-verified retained binaries. Record the new tool hashes when rebuilding.

Generate `util/job_launching/apps/define-trace-pipeline-validation.yml` from the retained `inputs/benchmark-definition.yml`, updating `exec_dir` and `data_dirs` to the new immutable inputs. The suite is `trace-pipeline-validation`; its three images each use arguments `1 2048 1 1 1 3`. The retained tracer is linked at `util/tracer_nvbit/tracer_tool/tracer_tool.so`, and the postprocessor at `util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing` in the task checkout. These paths must belong to this checkout or its own artifact namespace.

Run on an otherwise idle target GPU:

```bash
python3 "$NEW_RUN/validate_gpu_batches.py"
```

The script rotates sequential, overlapped and overlapped/uncompressed settings over five repetitions, preserves all raw data and per-phase receipts, and checks every launch output. It creates run-owned `traces` links into the selected trace directory. It raises an error if its output directories already exist, preserving previous observations. It does not pause or stop any other collector.

## Independent record checks

Build `verify_tracez.cpp` with zstd headers/libraries, mounting the checkout at `/source` if using the retained build command. For each raw kernel, decompress `.trace.xz` to a temporary text file (or use its `.trace` file), then run:

```bash
./verify_tracez raw-kernel.txt grouped-kernel.tracez
```

Run `python3 docs/experiments/trace-cpu-gpu-pipeline/verify_gpu_batches.py "$NEW_RUN"` after building the verifier at `$NEW_RUN/verify_tracez` to check every retained output and kernel.

The verifier is scoped to this experiment's one-CTA/one-warp inputs. It checks all instruction strings and the tracez index/subchunk counts. The complete retained verification JSON includes all 180 raw/grouped file hashes and commands. Replay regression commands, configurations and actual outputs are preserved in the text/tracez equivalence capsules and linked report data.

## Figures

```bash
python3 docs/experiments/trace-cpu-gpu-pipeline/analyze_gpu_batches.py "$NEW_RUN"
```

This reads the saved `GPU-BATCHES.json`, emits every batch value and min/median/max summary, and renders the plot without recollection. The original trace comparison is trace preservation and cycle-model regression; only real hardware runs measure collection throughput.
