# 📘 Using the Tracer Tool for Accel-Sim

This document explains how to use the `tracer_tool` for generating instruction traces for GPU applications. The tool supports full benchmark suites, individual applications, specific kernel tracing, source line mapping, and more.

---

## 🛠️ Setup and Installation

Before using the tracer, make sure to install and build the required tools:

```bash
# Install NVBit
./install_nvbit.sh

# Compile tracer tools
./make
```

---

## ⚙️ Option 1: Trace a Full Benchmark Suite

Use this if you're tracing an entire benchmark suite (e.g., Rodinia):

```bash
./run_hw_trace.py -B rodinia-3.1 -D 0
```

- `-B`: Benchmark suite name  (app list can be found or defined in [this file](../job_launching/apps/define-all-apps.yml))
- `-D`: Hardware device ID (e.g., GPU 0)

📁 Traces will be stored in:  
`../../hw_run/traces/device-0/`

**This script handles trace generation, post-processing, and cleanup automatically.**  
See the [Trace File Structure](#-trace-file-structure) section for details on the output.

---

## ⚙️ Option 2: Trace an Individual Application

Use this approach if you want to trace a specific application binary (e.g., `vectoradd`):

```bash
export CUDA_VISIBLE_DEVICES=0
LD_PRELOAD=./tracer_tool/tracer_tool.so ./nvbit_release/test-apps/vectoradd/vectoradd
```

📁 Traces will appear in the `traces/` folder.  
See the [Trace File Structure](#-trace-file-structure) section for contents.  
🔄 **Note:** Unlike Option 1, **you must manually perform post-processing**:

```bash
./tracer_tool/traces-processing/post-traces-processing ./traces/kernelslist
```

This will generate `.traceg` and `kernelslist.g` files that are ready for Accel-Sim.

---

## Overlap GPU capture and CPU postprocessing

Set `RUN_ROOT` to a fresh task-owned run directory. The launcher can start the
next GPU capture while CPU workers group and compress completed trace files:

```bash
./run_hw_trace.py -B rodinia-3.1 -D 0 \
  --postprocess-workers 1 --max-pending-postprocess 2 \
  --postprocess-threads 2 --keep-raw --results-dir "$RUN_ROOT"
```

`--postprocess-workers 0` is the default and preserves serial execution. One
launcher owns one GPU capture stream. Each completed capture is submitted to a
CPU worker; a bundle becomes ready for simulation only after that worker exits
successfully. `--max-pending-postprocess` bounds running and queued CPU bundles,
so GPU admission waits when the CPU queue reaches that limit. Its default is
twice the worker count. `--postprocess-threads` passes the existing `-j` option to
each postprocessor; it does not set a process-wide CPU affinity limit.

The generated scripts are separate:

- `run_capture.sh` runs the application under NVBit.
- `run_postprocess.sh` groups completed traces and writes replay files. It has no
  visible CUDA devices and removes CUDA injection/preload variables.
- `run.sh` runs both scripts serially for manual use.

The launcher records per-phase stdout, stderr and JSON receipts containing the
command, exit status and host timestamps. A failed capture does not submit CPU
work. CPU failures produce a nonzero launcher exit. Submitted workers finish
before the launcher exits, and a repeated run directory waits for its previous
worker before any files are reused. Raw files are retained on a reported
postprocessor failure; use `--keep-raw` to retain them on success as well.

`--results-dir` selects the run directories rather than the default `hw_run`
location. If traces and logs use separate disks, create each run directory's
`traces` link to its task-owned trace directory before launch. The validation
script preserves an example of this layout. Do not point a run at an active
bundle owned by another collector.

Raw xz compression can also be removed from the live capture with the tracer's
existing environment option:

```bash
TRACE_FILE_COMPRESS=0 ./run_hw_trace.py -B rodinia-3.1 -D 0 \
  --postprocess-workers 1 --postprocess-threads 2 \
  --max-pending-postprocess 2 --results-dir "$RUN_ROOT"
```

In that form, live capture writes uncompressed raw text, and the CPU worker
writes the compressed `.tracez` replay files. The default replay format remains
`tracez`; `--postprocess-format text` selects the existing `.traceg` format.
Uncompressed raw files need more temporary storage. Neither option moves GPU
instrumentation, channel draining or raw text formatting out of live capture.
The finite NVBit channel must still be drained while the kernel executes.

See the [source plan](../../docs/plans/trace-cpu-gpu-pipeline/PLAN.md) and
[validation report](../../docs/reports/trace-cpu-gpu-pipeline.md) for the exact
boundary, commands, equivalence checks and measured throughput. These host
collection timings are separate from native GPU kernel performance.

---

## 📦 Trace File Structure

Each trace folder contains:

- `kernel-*.trace`: Raw trace files (one per kernel)
- `kernelslist`: List of traced kernels and CUDA memcpy operations
- `stats.csv`: Summary statistics (instruction counts, kernel IDs, etc.)

After post-processing:
- `.traceg`: Grouped trace files by thread block
- `kernelslist.g`: Final trace list for use with Accel-Sim

---

## 🎯 Selective Tracing (Kernel-Based Filtering)

You can now trace specific kernels using the `DYNAMIC_KERNEL_RANGE` environment variable.

### Usage

Set `DYNAMIC_KERNEL_RANGE` to specify which kernel IDs (and optionally names) to trace. Supported formats:

- **Single ID:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="3"
  ```
  Traces only kernel 3.

- **Range:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="5-8"
  ```
  Traces kernels 5 through 8 (inclusive).

- **Open-ended Range:**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="10-"
  ```
  Traces from kernel 10 onward.

- **Multiple Ranges (space-separated):**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="2 5-8 10-"
  ```

- **With Name Filters (regex):**  
  ```bash
  export DYNAMIC_KERNEL_RANGE="5-8@kernel_a.*,kernel_b.*"
  ```
  Traces kernels 5–8 *only* if their names match `kernel_a.*` or `kernel_b.*`.

### To Disable Tracing But Still List Kernels

To list kernel metadata in `stats.csv` without generating traces, we set the DYNAMIC_KERNEL_RANGE to very large number  :
```bash
export DYNAMIC_KERNEL_RANGE="1000000"
```

This is useful for discovering kernel IDs and names without producing large trace files.


---

## ⏱️ Alternative: Trace Specific Code Regions with CUDA Profiling

Wrap regions to trace using CUDA APIs:
```cpp
cudaProfilerStart();
// region to trace
cudaProfilerStop();
```

Then disable default tracing:
```bash
export ACTIVE_FROM_START=0
```

---

## 🔍 Trace Source Line Mapping

Enable source line information in your traces:

1. Set environment variable:
```bash
export TRACE_LINEINFO=1
```

2. Rebuild benchmark applications:
```bash
source ./gpu-app-collection/src/setup_environment
make -j -C ./gpu-app-collection/src rodinia_2.0-ft
```

Traces will now include line number info from the original CUDA source (requires `-lineinfo` flag in NVCC).

---

## 📄 Trace Format Explanation

Each instruction has at least 10 required columns:

```
[line_num] PC mask dest_num [reg_dests] opcode src_num [reg_srcs] mem_width [addresscompress?] [mem_addresses]
```

Details:
- Fields in `[]` are optional and appear only if applicable.
- `dest_num = 0` → no destination register field.
- `mem_width = 0` → no memory address info present.

🧾 **Example:**
```
31 0 0 3 0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0
```

This line represents:
- Threadblock: (31, 0, 0)  
- PC: `0000`, Mask: `ffffffff`  
- One destination register: `R1`  
- Opcode: `IMAD.MOV.U32`  
- Two source registers: `R255`, `R255`  
- Not a memory instruction (`mem_width = 0`)
