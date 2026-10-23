/* Author1: Mahmoud Khairy, abdallm@purdue.com - 2019 */
/* Author2: Jason Shen, shen203@purdue.edu - 2019 */

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <algorithm> // for std::minmax_element
#include <bitset>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"
#include "formatting/trace_capture.h"

/* contains definition of the WarpsyncCollectiveWatchdog structure */
#include "watchdog.h"

/* for general utils functions */
#include "utils.h"

#define TRACER_VERSION "5"
#define TRACER_VERSION_ALLOW_REG_VAL "6"

static int get_attr_with_kernel_fallback(CUfunction func,
                                         CUfunction_attribute attr) {
  int value = 0;
  CUresult res = cuFuncGetAttribute(&value, attr, func);
  if (res == CUDA_ERROR_INVALID_HANDLE) {
    CUdevice dev = 0;
    if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
      cuKernelGetAttribute(&value, attr, (CUkernel)func, dev);
    }
  }
  return value;
}

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/*
 * Instance of the WarpsyncCollectiveWatchdog,
 * to be initialized in nvbit_at_init. This is a temporary solution and should
 * be removed once NVBit fixes the bug with warpsync.collective.
 */
std::unique_ptr<WatchdogInterface> warpsync_collective_watchdog{};
bool enable_watchdog = true;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
int lineinfo = 0;
bool skip_tma_mem = false;
bool allow_reg_val_tracing = true;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* Use xz to compress the *.trace file */
int xz_compress_trace = 0;
int defer_formatting = 0;

// Debugging helper
#define DPRINTF(fmt, ...)                                                      \
  {                                                                            \
    if (verbose > 0)                                                           \
      printf(fmt, ##__VA_ARGS__);                                              \
  }

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* Variable to control if NVBit instrumentation is enabled */
bool nvbit_instrumentation_enabled = 1;
const char *nvbit_instrumentation_tag = "DEFAULT";

/* NVBit instrumentation control functions in C so interaction
 * is easier with the tool*/
extern "C" {
void set_nvbit_instrumentation_tag(const char *tag) {
  nvbit_instrumentation_tag = tag;
}

void enable_nvbit_instrumentation() { nvbit_instrumentation_enabled = true; }

void disable_nvbit_instrumentation() { nvbit_instrumentation_enabled = false; }
}

std::string user_folder = getcwd(NULL, 0);
std::string cwd = getcwd(NULL, 0);
std::string traces_location = cwd + "/traces/";
std::string kernelslist_location = cwd + "/traces/kernelslist";
std::string stats_location = cwd + "/traces/stats.csv";

std::unordered_map<CUcontext, std::string> ctx_kernelslist;
std::unordered_map<CUcontext, std::string> ctx_stats_location;
std::unordered_map<CUcontext, int> ctx_kernelid;
std::unordered_map<CUcontext, FILE *> ctx_resultsFile;
std::unordered_map<CUcontext, std::unique_ptr<accelsim_trace::TraceFormatter>> ctx_formatters;
std::unordered_map<CUcontext, std::vector<FILE *>> ctx_deferred_files;
std::unordered_map<CUcontext, std::string> ctx_current_kernel_name;

std::string kernel_ranges = "";

struct KernelRange {
  uint64_t start;
  uint64_t end; // UINT64_MAX means open-ended
  std::vector<std::regex>
      kernel_name_regexes; // Vector of regexes for multiple patterns
};
std::vector<KernelRange> g_kernel_ranges;
uint64_t g_max_kernel_id = 0;
void parse_kernel_ranges_from_env() {
  g_kernel_ranges.clear();
  g_max_kernel_id = 0;

  const char *env_var = std::getenv("DYNAMIC_KERNEL_RANGE");
  DPRINTF("DYNAMIC_KERNEL_RANGE environment variable: %s\n", env_var);
  if (!env_var || std::string(env_var).empty()) {
    g_kernel_ranges.push_back({0, 0, {std::regex(".*")}}); // 0 end = trace all
    DPRINTF("No DYNAMIC_KERNEL_RANGE environment variable found, tracing all "
            "kernels\n");
    return;
  }
  std::string input(env_var);
  std::istringstream stream(input);
  std::string token;

  while (stream >> token) {
    if (token.empty())
      continue;

    uint64_t start = 0, end = 0;
    std::vector<std::regex> regexes;

    size_t at_pos = token.find('@');
    std::string range_part, regex_part;

    if (at_pos != std::string::npos) {
      range_part = token.substr(0, at_pos);
      regex_part = token.substr(at_pos + 1);
    } else {
      range_part = token;
    }

    // Parse the range
    if (!range_part.empty()) {
      size_t dash_pos = range_part.find('-');
      if (dash_pos != std::string::npos) {
        std::string start_str = range_part.substr(0, dash_pos);
        std::string end_str = range_part.substr(dash_pos + 1);

        start = std::stoull(start_str);
        if (!end_str.empty()) {
          end = std::stoull(end_str);
        } else {
          end = 0; // open-ended
        }
      } else {
        start = std::stoull(range_part);
        end = start;
      }
    } else {
      // No range → match all IDs
      start = 0;
      end = 0;
    }

    // Parse the regexes
    if (!regex_part.empty()) {
      std::istringstream regex_stream(regex_part);
      std::string regex_token;
      while (std::getline(regex_stream, regex_token, ',')) {
        try {
          regexes.emplace_back(regex_token);
        } catch (const std::regex_error &e) {
          std::cerr << "Invalid regex: " << regex_token << std::endl;
        }
      }
    } else {
      regexes.emplace_back(".*"); // match all kernel names
    }

    g_kernel_ranges.push_back({start, end, regexes});
    DPRINTF("Added kernel range: %lu-%lu with regexes: %s\n", start, end,
            regex_part.c_str());
    if (end > g_max_kernel_id) {
      g_max_kernel_id = end;
    }
  }
}

bool should_trace_kernel(uint64_t kernel_id, const std::string &kernel_name) {
  for (const auto &range : g_kernel_ranges) {
    // Check range for kernel ID
    if (range.end == 0) {
      if (kernel_id >= range.start) {
        // Match any of the regexes for this range
        for (const auto &regex : range.kernel_name_regexes) {
          if (std::regex_match(kernel_name, regex)) {
            return true;
          }
        }
      }
    } else if (kernel_id >= range.start && kernel_id <= range.end) {
      // Match any of the regexes for this range
      for (const auto &regex : range.kernel_name_regexes) {
        if (std::regex_match(kernel_name, regex)) {
          return true;
        }
      }
    }
  }
  return false;
}

/* File pointers for the kernels, and stats files */
static FILE *kernelsFile = NULL;
static FILE *statsFile = NULL;
static bool first_call = true;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

/* Spinlock handling mode: 0=none, 1=fast_forward, 2=mark_region */
const int SPINLOCK_MODE_NONE = 0;
const int SPINLOCK_MODE_FAST_FORWARD = 1;
const int SPINLOCK_MODE_MARK_REGION = 2;
int spinlock_handling_mode = 0;
int spinlock_iter_to_keep = 0;
// Map from kernel name to spinlock instruction indices
std::map<std::string, std::vector<uint32_t> *> spinlock_instr_map;
std::pair<std::string, std::vector<uint32_t>>
parse_spinlock_instructions(const std::string &line);

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(lineinfo, "TRACE_LINEINFO", 0,
              "Include source code line info at the start of each traced line. "
              "The target binary must be compiled with -lineinfo or "
              "--generate-line-info");
  GET_VAR_STR(
      kernel_ranges, "DYNAMIC_KERNEL_RANGE",
      "Specify kernel IDs or ranges to trace. Format:\n"
      "  - Single ID:       \"2\" traces only kernel 2.\n"
      "  - Range:           \"5-8\" traces kernels 5 through 8 (inclusive).\n"
      "  - Open-ended:      \"10-\" traces from kernel 10 onward.\n"
      "  - Multiple ranges: \"2 5-8 10-\" (space-separated).\n"
      "  - With regex:      \"5-8@kernel_a.*,kernel_b.*\" traces kernels 5–8 "
      "with matching names.\n"
      "If unset or empty, all kernels will be traced from the beginning.");
  GET_VAR_INT(
      active_from_start, "ACTIVE_FROM_START", 1,
      "Start instruction tracing from start or wait for cuProfilerStart "
      "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_RANGE options have no "
      "effect");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(enable_compress, "TOOL_COMPRESS", 1, "Enable traces compression");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(
      terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT",
      0, "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0,
              "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(xz_compress_trace, "TRACE_FILE_COMPRESS", 1,
              "Create xz-compressed trace"
              "file");
  GET_VAR_INT(defer_formatting, "TRACE_DEFER_FORMATTING", 0,
              "Collect GPU records; format and compress them in a separate CPU phase");
  GET_VAR_INT(spinlock_handling_mode, "SPINLOCK_HANDLING_MODE", 0,
              "Spinlock handling mode: 0=none, 1=fast_forward, 2=mark_region");
  GET_VAR_INT(spinlock_iter_to_keep, "SPINLOCK_ITER_TO_KEEP", 1,
              "Number of iterations to keep for spinlock handling");
  GET_VAR_INT(enable_watchdog, "ENABLE_WATCHDOG", 1,
              "Enable the watchdog to skip instructions between "
              "WARPSYNC.COLLECTIVE and its target instruction (inclusive)");
  GET_VAR_INT(skip_tma_mem, "SKIP_TMA_MEM", 0,
              "Enable the skipping of TMA memory instructions");
  GET_VAR_INT(allow_reg_val_tracing, "ALLOW_REG_VAL_TRACING", 1,
              "EXPERIMENTAL: Enable the tracing of register values. Trace "
              "format is not stable. Trace version is 6.");
  GET_VAR_INT(
      nvbit_instrumentation_enabled, "NVBIT_INSTRUMENTATION_ENABLED", 1,
      "Enable NVBit instrumentation. Can be controlled at runtime using "
      "enable_nvbit_instrumentation() and disable_nvbit_instrumentation()");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  active_region = false;
  char *usr_defined_folder = std::getenv("TRACES_FOLDER");
  if (usr_defined_folder != NULL)
    user_folder = usr_defined_folder;
  parse_kernel_ranges_from_env();

  // Read in the spinlock_instructions.txt and build a map from kernel name to
  // spinlock instruction indices
  if (spinlock_handling_mode > SPINLOCK_MODE_NONE) {
    std::string spinlock_instr_file =
        user_folder + "/spinlock_detection/spinlock_instructions.txt";
    std::ifstream instr_fs(spinlock_instr_file);
    std::string line;
    while (std::getline(instr_fs, line)) {
      // Skip blank lines and comment lines emitted by the spinlock tool
      if (line.empty() || line[0] == '#')
        continue;
      auto [kernel_name, indices] = parse_spinlock_instructions(line);
      spinlock_instr_map[kernel_name] = new std::vector<uint32_t>(indices);
    }
    instr_fs.close();
  }
  warpsync_collective_watchdog = WatchdogFactory::create(enable_watchdog);
}

/* Set used to avoid re-instrumenting the same functions multiple times.
 * Keyed on the CUfunction handle, the function's PC address, and its mangled
 * name, since a single program (e.g. cuFFT LTO) can reuse a handle or a PC
 * address for a different kernel across launches. */
std::unordered_set<std::tuple<CUfunction, uint64_t, std::string>,
                   accelsimUtils::TupleHash>
    already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    uint64_t func_addr = nvbit_get_func_addr(ctx, f);
    std::string func_name = nvbit_get_func_name(ctx, f, true);
    if (!already_instrumented.insert(std::make_tuple(f, func_addr, func_name))
             .second) {
      continue;
    }

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose >= 1) {
      printf("Inspecting function %s at address 0x%lx\n",
             nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    // reset the watchdog
    warpsync_collective_watchdog->reset();

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      uint32_t line_num = 0;

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      if (verbose >= 2) {
        instr->printDecoded();
      }

      warpsync_collective_watchdog->observe_instruction(instr);

      if (warpsync_collective_watchdog->is_in_region()) {
        // if the opcode matches with "WARPSYNC.COLLECTIVE",
        // print to stdout the PC value and the immeidate value
        if (verbose > 1 &&
            strcmp(instr->getOpcode(), "WARPSYNC.COLLECTIVE") == 0) {
          assert(1 < instr->getNumOperands());
          printf(
              "Encountered WARPSYNC.COLLECTIVE PC: 0x%lx, Immediate: 0x%lx\n",
              instr->getOffset(), instr->getOperand(1)->u.imm_uint64.value);
        }
        continue;
      }

      if (skip_tma_mem && instr->isTMAMem()) {
        continue;
      }

      if (lineinfo) {
        char *file_name, *dir_name;
        nvbit_get_line_info(ctx, func, instr->getOffset(), &file_name,
                            &dir_name, &line_num);
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];

      /* check all operands. For now, we ignore constant, TEX, predicates and
       * unified registers. We only report vector regisers */
      int src_oprd[MAX_SRC];
      int srcNum = 0;
      int dst_oprd = -1;
      int mem_oper_idx = -1;
      int num_mref = 0;
      // Support up to two immediate operands
      uint64_t imm_value = 0;
      uint64_t imm_value2 = 0;
      int imm_count = 0;
      // Check if `gsb` exists in full SASS string and the instruction is a GMMA
      // instruction
      bool is_gmma_instruction = strstr(instr->getSass(), "GMMA") != NULL;
      // A GMMA instruction with gsb is a commit group
      int is_gmma_commit_group =
          is_gmma_instruction && strstr(instr->getSass(), "gsb") != NULL;

      for (int i = 0; i < instr->getNumOperands(); ++i) {
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          assert(srcNum < MAX_SRC);
          src_oprd[srcNum] = instr->getOperand(i)->u.mref.ra_num;
          srcNum++;
          mem_oper_idx++;
          num_mref++;
          // if(mem_oper_idx == 0){
          //   mem_oper_idx = 1; // loop control
          // }
        } else if (op->type == InstrType::OperandType::REG) {
          if (i == 0) {
            // find dst reg
            dst_oprd = instr->getOperand(0)->u.reg.num;
          } else {
            // find src regs
            assert(srcNum < MAX_SRC);
            src_oprd[srcNum] = instr->getOperand(i)->u.reg.num;
            srcNum++;
          }
        } else if (op->type == InstrType::OperandType::UREG) {
          if (i == 0) {
            // find dst reg
            dst_oprd = instr->getOperand(0)->u.reg.num + UREG_OFFSET;
          } else {
            // find src regs
            assert(srcNum < MAX_SRC);
            src_oprd[srcNum] = instr->getOperand(i)->u.reg.num + UREG_OFFSET;
            srcNum++;
          }
        }
        // Add immediate value(s) for instructions with IMM_UINT64 operands
        else if (op->type == InstrType::OperandType::IMM_UINT64) {
          if (imm_count == 0) {
            imm_value = instr->getOperand(i)->u.imm_uint64.value;
          } else if (imm_count == 1) {
            imm_value2 = instr->getOperand(i)->u.imm_uint64.value;
          }
          imm_count++;
        }
        // TMA param handle is handled as a whole with
        // nvbit_add_call_arg_tma_param_handle
        // Skip here
      }

      do {
        /* insert call to the instrumentation function with its
         * arguments */
        nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);

        /* pass predicate value */
        nvbit_add_call_arg_guard_pred_val(instr);

        /* send opcode and pc */
        nvbit_add_call_arg_const_val32(instr, opcode_id);
        nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());

        /* tma setup */
        if (instr->isTMAMem()) {
          // Set is_mem to true and add tma param handle
          nvbit_add_call_arg_const_val32(instr, 1);
          nvbit_add_call_arg_tma_param_handle_and_size(instr, ctx);
        } else {
          // Dummy values for regular instructions
          nvbit_add_call_arg_const_val32(instr, 0);
          nvbit_add_call_arg_const_val64(instr, 0);
          nvbit_add_call_arg_const_val32(instr, 0);
        }

        /* mem addresses info */
        if (mem_oper_idx >= 0) {
          // Set is_mem to true
          nvbit_add_call_arg_const_val32(instr, 1);
          assert(num_mref <= 2);
          if (num_mref == 2) { // LDGSTS
            nvbit_add_call_arg_mref_addr64(instr, 1 - mem_oper_idx);
          } else {
            nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
          }
          nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
        } else {
          nvbit_add_call_arg_const_val32(instr, 0);
          nvbit_add_call_arg_const_val64(instr, static_cast<uint64_t>(-1));
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }

        /* reg info */
        nvbit_add_call_arg_const_val32(instr, dst_oprd);
        for (int i = 0; i < srcNum; i++) {
          nvbit_add_call_arg_const_val32(instr, src_oprd[i]);
        }
        for (int i = srcNum; i < MAX_SRC; i++) {
          nvbit_add_call_arg_const_val32(instr, static_cast<uint32_t>(-1));
        }
        nvbit_add_call_arg_const_val32(instr, srcNum);

        /* immediate info (up to two immediates) */
        nvbit_add_call_arg_const_val64(instr, imm_value);
        nvbit_add_call_arg_const_val64(instr, imm_value2);

        /* add pointer to channel_dev and other counters*/
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
        nvbit_add_call_arg_const_val64(instr,
                                       (uint64_t)&total_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(
            instr, (uint64_t)&reported_dynamic_instr_counter);
        nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report);
        /* Add Source code line number for current instr */
        nvbit_add_call_arg_const_val32(instr, (int)line_num);
        /* Add instruction index for current instr (spinlock detection) */
        nvbit_add_call_arg_const_val32(instr, (uint32_t)instr->getIdx());

        /* add register values */
        // If >= 256, it is a uniform register
        auto add_reg_val = [&](int reg_num) {
          if (reg_num >= 256) {
            nvbit_add_call_arg_ureg_val(instr, reg_num - UREG_OFFSET);
          } else {
            nvbit_add_call_arg_reg_val(instr, reg_num);
          }
        };
        if (dst_oprd >= 0) {
          add_reg_val(dst_oprd);
        } else {
          add_reg_val(0);
        }
        for (int i = 0; i < srcNum; i++) {
          add_reg_val(src_oprd[i]);
        }
        for (int i = srcNum; i < MAX_SRC; i++) {
          // Let tool read in dummy values
          // this will get ignored anyway when
          // we are printing to files by checking src_oprd's reg number
          add_reg_val(0);
        }

        // Opcode-defined generic aux register value (per-lane).
        // Default: 0 (unused). Add opcode-specific cases below.
        //   SYNCS.EXCH.64 — hi half (Rn+1) of src2, to be fused with
        //   srcRegVals[1] into a 64-bit column by the writer.
        int aux_reg = 0;
        if (strcmp(instr->getOpcode(), "SYNCS.EXCH.64") == 0) {
          assert(srcNum >= 2 && "SYNCS.EXCH.64 expected >= 2 src regs");
          assert(src_oprd[1] != 255 && src_oprd[1] != 511 &&
                 "SYNCS.EXCH.64 src2 is RZ/URZ; cannot form 64-bit pair");
          aux_reg = src_oprd[1] + 1;
        }
        add_reg_val(aux_reg);

        // Add is_gmma_commit_group flag
        nvbit_add_call_arg_const_val32(instr, is_gmma_commit_group);
        mem_oper_idx--;
      } while (mem_oper_idx >= 0);

      cnt++;
    }
  }
}

__global__ void flush_channel() {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  channel_dev.push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  channel_dev.flush();
}

static accelsim_trace::FormattingOptions formatting_options(const std::string &kernel_name) {
  accelsim_trace::FormattingOptions options;
  options.address_compression=enable_compress;
  options.print_core_id=print_core_id;
  options.lineinfo=lineinfo;
  options.register_values=allow_reg_val_tracing;
  options.raw_xz_compression=xz_compress_trace;
  options.spinlock_mode=spinlock_handling_mode;
  options.spinlock_iterations=spinlock_iter_to_keep;
  auto found=spinlock_instr_map.find(kernel_name);
  options.spinlock_metadata_present=found!=spinlock_instr_map.end() && found->second;
  if(options.spinlock_metadata_present) options.spinlock_indices=*found->second;
  return options;
}

// The SDK's TMA APIs take the live CUDA context. Collect their returned values
// here; their textual formatting and address compression run in the CPU phase.
static accelsim_trace::TmaSnapshot capture_tma(CUcontext ctx,inst_trace_t *trace,
                                               const std::string &opcode) {
  accelsim_trace::TmaSnapshot captured;
  if(trace->inst_type!=INST_TMA || !(trace->active_mask & trace->predicate_mask))
    return captured;
  const char *name=opcode.c_str();
  TMATransferInfo_t info=nvbit_parse_tma_transfer_info(ctx,name,
      trace->inst.tma.tma_param_handle,trace->inst.tma.tma_param_handle_size);
  captured.transfer_size=info.transfer_size;
  captured.destination_is_shared=info.dst_memspace==InstrType::MemorySpace::DISTRIBUTED_SHARED;
  if(captured.destination_is_shared) {
    captured.mbar_valid=info.dst.shared.is_mbar_valid;
    captured.mbar_address=info.dst.shared.mbar_address;
    captured.multicast=info.is_multicast;
    captured.multicast_mask=info.multicast_cta_mask;
  }
  captured.byte_count=info.byte_count;
  captured.oob_byte_count=info.is_tensor ? info.tensor.oob_transfer_count*info.transfer_size : 0;
  captured.transfer_count=info.transfer_count;
  struct Collector {
    std::vector<uint64_t> *addresses;
    static void callback(const TMAElementAddress_t *addresses,size_t count,void *argument) {
      auto *self=static_cast<Collector *>(argument);
      for(size_t i=0;i<count;++i)
        if(!addresses[i].is_oob) self->addresses->push_back(addresses[i].address);
    }
  } collector{&captured.global_addresses};
  if(info.src_memspace==InstrType::MemorySpace::GLOBAL)
    nvbit_parse_tma_src_addrs(ctx,name,trace->inst.tma.tma_param_handle,
        trace->inst.tma.tma_param_handle_size,Collector::callback,&collector);
  else if(info.dst_memspace==InstrType::MemorySpace::GLOBAL)
    nvbit_parse_tma_dst_addrs(ctx,name,trace->inst.tma.tma_param_handle,
        trace->inst.tma.tma_param_handle_size,Collector::callback,&collector);
  return captured;
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                nvbit_api_cuda_t cbid, void *params,
                                bool stream_capture = false,
                                bool build_graph = false) {
  // no need to sync during stream capture or manual graph build, since no
  // kernel is actually launched.
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
  }

  // Mark if the kernel should be traced
  std::string func_name = std::string(nvbit_get_func_name(ctx, func, true));
  if (active_from_start && should_trace_kernel(ctx_kernelid[ctx], func_name))
    active_region = nvbit_instrumentation_enabled;
  else
    active_region = false;

  // Terminate tracing if the limit number of kernels is reached
  if (terminate_after_limit_number_of_kernels_reached && g_max_kernel_id != 0 &&
      ctx_kernelid[ctx] > g_max_kernel_id) {
    exit(0);
  }

  // Get launch config for this kernel
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  CUstream hStream;
  if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
      cbid == API_CUDA_cuLaunchKernelEx) {
    cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
    gridDimX = p->config->gridDimX;
    gridDimY = p->config->gridDimY;
    gridDimZ = p->config->gridDimZ;
    blockDimX = p->config->blockDimX;
    blockDimY = p->config->blockDimY;
    blockDimZ = p->config->blockDimZ;
    sharedMemBytes = p->config->sharedMemBytes;
    hStream = p->config->hStream;
  } else {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
    gridDimX = p->gridDimX;
    gridDimY = p->gridDimY;
    gridDimZ = p->gridDimZ;
    blockDimX = p->blockDimX;
    blockDimY = p->blockDimY;
    blockDimZ = p->blockDimZ;
    sharedMemBytes = p->sharedMemBytes;
    hStream = p->hStream;
  }

  // Get the number of registers and shared memory size for the kernel
  int nregs = get_attr_with_kernel_fallback(func, CU_FUNC_ATTRIBUTE_NUM_REGS);
  int shmem_static_nbytes =
      get_attr_with_kernel_fallback(func, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
  int binary_version =
      get_attr_with_kernel_fallback(func, CU_FUNC_ATTRIBUTE_BINARY_VERSION);

  // Instrument the kernel if needed
  instrument_function_if_needed(ctx, func);

  // Enable or disable tracing based on the active region
  if (active_region) {
    nvbit_enable_instrumented(ctx, func, true);
    stop_report = false;
  } else {
    nvbit_enable_instrumented(ctx, func, false);
    stop_report = true;
  }

  // Create the trace file per kernel
  char buffer[2048];
  std::string trace_filename = user_folder + "/traces/";
  sprintf(buffer, "%s/kernel-%d-ctx_0x%lx.trace", trace_filename.c_str(),
          ctx_kernelid[ctx], ctx);

  if (!stop_report) {
    if (defer_formatting) {
      std::string records=std::string(buffer)+".records";
      ctx_resultsFile[ctx]=fopen(records.c_str(),"wb");
      if(!ctx_resultsFile[ctx]) throw std::runtime_error("Cannot create "+records);
      ctx_deferred_files[ctx].push_back(ctx_resultsFile[ctx]);
      printf("Collecting GPU records to %s\n",records.c_str());
    } else if (!xz_compress_trace) {
      ctx_resultsFile[ctx] = fopen(buffer, "w");
      printf("Writing results to %s\n", buffer);
    } else {
      char cmd_buffer[1039];
      sprintf(cmd_buffer, "xz -1 -T0 > %s.xz", buffer);
      ctx_resultsFile[ctx] = popen(cmd_buffer, "w");
      printf("Writing results to %s.xz\n", buffer);
    }

    // Writing header information
    fprintf(ctx_resultsFile[ctx], "-kernel name = %s\n",
            nvbit_get_func_name(ctx, func, true));
    fprintf(ctx_resultsFile[ctx], "-kernel id = %d\n", ctx_kernelid[ctx]);
    fprintf(ctx_resultsFile[ctx], "-grid dim = (%d,%d,%d)\n", gridDimX,
            gridDimY, gridDimZ);
    fprintf(ctx_resultsFile[ctx], "-block dim = (%d,%d,%d)\n", blockDimX,
            blockDimY, blockDimZ);
    fprintf(ctx_resultsFile[ctx], "-shmem = %d\n",
            shmem_static_nbytes + sharedMemBytes);
    fprintf(ctx_resultsFile[ctx], "-nregs = %d\n", nregs);
    fprintf(ctx_resultsFile[ctx], "-binary version = %d\n", binary_version);
    fprintf(ctx_resultsFile[ctx], "-cuda stream id = %lu\n", (uint64_t)hStream);
    fprintf(ctx_resultsFile[ctx], "-shmem base_addr = 0x%016lx\n",
            (uint64_t)nvbit_get_shmem_base_addr(ctx));
    fprintf(ctx_resultsFile[ctx], "-local mem base_addr = 0x%016lx\n",
            (uint64_t)nvbit_get_local_mem_base_addr(ctx));
    fprintf(ctx_resultsFile[ctx], "-nvbit version = %s\n", NVBIT_VERSION);
    fprintf(ctx_resultsFile[ctx], "-accelsim tracer version = %s\n",
            allow_reg_val_tracing ? TRACER_VERSION_ALLOW_REG_VAL
                                  : TRACER_VERSION);
    fprintf(ctx_resultsFile[ctx], "-enable lineinfo = %d\n", lineinfo);
    fprintf(ctx_resultsFile[ctx], "\n");

    if (allow_reg_val_tracing) {
      fprintf(ctx_resultsFile[ctx],
              "#traces format = [line_num] PC mask dest_num [reg_dests] "
              "opcode src_num "
              "[reg_srcs] mem_width [adrrescompress?] [mem_addresses] "
              "immediate1 immediate2 Val|NoVal [dest_val_num] [dest_val] ... "
              "[src_val1_num] [src_val1] ... [...] "
              "[src_valX_num] [src_valX] ...\n");
    } else {
      fprintf(ctx_resultsFile[ctx],
              "#traces format = [line_num] PC mask dest_num [reg_dests] "
              "opcode src_num "
              "[reg_srcs] mem_width [adrrescompress?] [mem_addresses] "
              "immediate1 immediate2\n");
    }
    fprintf(ctx_resultsFile[ctx], "\n");
    auto options=formatting_options(func_name);
    if(defer_formatting)
      accelsim_trace::begin_capture(ctx_resultsFile[ctx],options);
    else
      ctx_formatters[ctx]=std::make_unique<accelsim_trace::TraceFormatter>(ctx_resultsFile[ctx],options);
  }

  kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
  // This will be a relative path to the traces file

  sprintf(buffer, "kernel-%d-ctx_0x%lx.trace%s", ctx_kernelid[ctx], ctx,
          xz_compress_trace ? ".xz" : "");
  if (!stop_report) {
    fprintf(kernelsFile, buffer);
    fprintf(kernelsFile, "\n");
  }
  fclose(kernelsFile);

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  unsigned blocks = gridDimX * gridDimY * gridDimZ;
  unsigned threads = blockDimX * blockDimY * blockDimZ;

  fprintf(statsFile, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", buffer,
          nvbit_get_func_name(ctx, func, true), gridDimX, gridDimY, gridDimZ,
          blocks, blockDimX, blockDimY, blockDimZ, threads);

  fclose(statsFile);

  ctx_kernelid[ctx]++;
  ctx_current_kernel_name[ctx] =
      std::string(nvbit_get_func_name(ctx, func, true));
  recv_thread_receiving = true;
}

static void leave_kernel_launch(CUcontext ctx, CUfunction func) {
  /* make sure current kernel is completed */
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("cuda error: %s\n", cudaGetErrorName(err));
    printf("cuda error explanation: %s\n", cudaGetErrorString(err));
    if (defer_formatting || !xz_compress_trace) {
      fclose(ctx_resultsFile[ctx]);
    } else {
      pclose(ctx_resultsFile[ctx]);
    }
  }
  assert(err == cudaSuccess);

  /* make sure we prevent re-entry on the nvbit_callback when issuing
   * the flush_channel kernel */
  skip_flag = true;

  /* issue flush of channel so we are sure all the memory accesses
   * have been pushed */
  flush_channel<<<1, 1>>>();
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  /* unset the skip flag */
  skip_flag = false;

  /* wait here until the receiving thread has not finished with the
   * current kernel */
  while (recv_thread_receiving) {
    pthread_yield();
  }

  unsigned total_insts_per_kernel =
      total_dynamic_instr_counter - old_total_insts;
  old_total_insts = total_dynamic_instr_counter;

  unsigned reported_insts_per_kernel =
      reported_dynamic_instr_counter - old_total_reported_insts;
  old_total_reported_insts = reported_dynamic_instr_counter;

  statsFile = fopen(ctx_stats_location[ctx].c_str(), "a");
  fprintf(statsFile, "%d,%d", total_insts_per_kernel,
          reported_insts_per_kernel);
  fprintf(statsFile, "\n");
  fclose(statsFile);

  if (!stop_report) {
    if (defer_formatting) {
      FILE *file=ctx_resultsFile[ctx];
      accelsim_trace::finish_capture(file,id_to_opcode_map);
      if(fclose(file)) throw std::runtime_error("Closing collected GPU records failed");
      auto &files=ctx_deferred_files[ctx];
      files.erase(std::remove(files.begin(),files.end(),file),files.end());
    } else if (!xz_compress_trace) {
      fclose(ctx_resultsFile[ctx]);
    } else {
      pclose(ctx_resultsFile[ctx]);
    }
  }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  if (skip_flag)
    return;

  if (first_call == true) {
    first_call = false;
    std::string traces_folder = user_folder + "/traces";
    if (mkdir(traces_folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ==
        -1) {
      if (errno == EEXIST) {
        // alredy exists
      } else {
        // something else
        std::cout << "cannot create folder error:" << strerror(errno)
                  << std::endl;
        return;
      }
    }
    kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "w");
    statsFile = fopen(ctx_stats_location[ctx].c_str(), "w");
    fprintf(statsFile,
            "kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    fclose(statsFile);
  }

  switch (cbid) {
  // We start with recording kernel launch events
  // Identify all the possible CUDA launch events without stream
  // parameters, they will not get involved with cuda graph
  case API_CUDA_cuLaunch:
  case API_CUDA_cuLaunchGrid: {
    cuLaunch_params *p = (cuLaunch_params *)params;
    CUfunction func = p->f;
    if (!is_exit) {
      enter_kernel_launch(ctx, func, cbid, params, false, false);
    } else {
      leave_kernel_launch(ctx, func);
    }
  } break;
  // To support kernel launched by cuda graph (in addition to existing kernel
  // launche method), we need to do:
  //
  // 1. instrument kernels at cudaGraphAddKernelNode event. This is for cases
  // that kernels are manually added to a cuda graph.
  // 2. distinguish captured kernels when kernels are recorded to a graph
  // using stream capture. cudaStreamIsCapturing() tells us whether a stream
  // is capturiong.
  // 3. per-kernel instruction counters, since cuda graph can launch multiple
  // kernels at the same time.
  //
  // Three cases:
  //
  // 1. original kernel launch:
  //     1a. for any kernel launch without using a stream, we instrument it
  //     before it is launched, call cudaDeviceSynchronize after it is
  //     launched and read the instruction counter of the kernel.
  //     1b. for any kernel launch using a stream, but the stream is not
  //     capturing, we do the same thing as 1a.
  //
  //  2. cuda graph using stream capturing: if a kernel is launched in a
  //  stream and the stream is capturing. We instrument the kernel before it
  //  is launched and do nothing after it is launched, because the kernel is
  //  not running until cudaGraphLaunch. Instead, we issue a
  //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
  //  instruction counters, since a cloned graph might be launched afterwards.
  //
  //  3. cuda graph manual: we instrument the kernel added by
  //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch as 2.
  //
  // The above method should handle most of cuda graph launch cases.
  // kernel launches with stream parameter, they can be used for cuda graph
  case API_CUDA_cuLaunchKernel_ptsz:
  case API_CUDA_cuLaunchKernel:
  case API_CUDA_cuLaunchCooperativeKernel:
  case API_CUDA_cuLaunchCooperativeKernel_ptsz:
  case API_CUDA_cuLaunchKernelEx:
  case API_CUDA_cuLaunchKernelEx_ptsz:
  case API_CUDA_cuLaunchGridAsync: {
    CUfunction func;
    CUstream hStream;

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
        cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
      hStream = p->config->hStream;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
               cbid == API_CUDA_cuLaunchKernel ||
               cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
               cbid == API_CUDA_cuLaunchCooperativeKernel) {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      func = p->f;
      hStream = p->hStream;
    } else {
      cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
      func = p->f;
      hStream = p->hStream;
    }

    cudaStreamCaptureStatus streamStatus;
    /* check if the stream is capturing, if yes, do not sync */
    CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
    if (!is_exit) {
      bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
      enter_kernel_launch(ctx, func, cbid, params, stream_capture);
    } else {
      if (streamStatus != cudaStreamCaptureStatusActive) {
        if (verbose >= 1) {
          printf("kernel %s not captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
        leave_kernel_launch(ctx, func);
      } else {
        if (verbose >= 1) {
          printf("kernel %s captured by cuda graph\n",
                 nvbit_get_func_name(ctx, func));
        }
      }
    }
  } break;
  case API_CUDA_cuGraphAddKernelNode: {
    cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
    CUfunction func = p->nodeParams->func;

    if (!is_exit) {
      // cuGraphAddKernelNode_params->nodeParams is the same as
      // cuLaunchKernel_params up to sharedMemBytes
      enter_kernel_launch(ctx, func, cbid, (void *)p->nodeParams, false, true);
    }
  } break;
  case API_CUDA_cuGraphLaunch: {
    // if we are exiting a cuda graph launch:
    // Wait until the graph is completed using
    // cudaStreamSynchronize()
    if (is_exit) {
      cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      assert(cudaGetLastError() == cudaSuccess);
      /* push a flush channel kernel */
      skip_flag = true;
      flush_channel<<<1, 1, 0, p->hStream>>>();
      CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
      skip_flag = false;
      assert(cudaGetLastError() == cudaSuccess);
    }

  } break;
  // Now we need to record cuda memcpy events
  case API_CUDA_cuMemcpyHtoD_v2: {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      char buffer[1024];
      kernelsFile = fopen(ctx_kernelslist[ctx].c_str(), "a");
      sprintf(buffer, "MemcpyHtoD,0x%016llx,%llu", p->dstDevice, p->ByteCount);
      fprintf(kernelsFile, buffer);
      fprintf(kernelsFile, "\n");
      fclose(kernelsFile);
    }
  } break;
  // For cuProfiler, we need to set the active region accordingly
  case API_CUDA_cuProfilerStart: {
    if (is_exit && !active_from_start) {
      active_region = nvbit_instrumentation_enabled;
    }
  } break;
  case API_CUDA_cuProfilerStop: {
    if (is_exit && !active_from_start) {
      active_region = false;
    }
  } break;
  default:
    break;
  }
}

void trim_string(std::string &str) {
  // Remove the leading and trailing spaces
  str.erase(0, str.find_first_not_of(' '));
  str.erase(str.find_last_not_of(' ') + 1);
}

std::pair<std::string, std::vector<uint32_t>>
parse_spinlock_instructions(const std::string &line) {
  std::vector<uint32_t> indices;
  // Each line is of the form: <kernel_id>, <kernel_name>: <indices>
  // Though kernel id is not used
  // -2 for the comma and the space
  size_t name_length = line.find(':') - line.find(',') - 2;
  std::string kernel_name = line.substr(line.find(',') + 2, name_length);
  std::string indices_str = line.substr(line.find(':') + 1);
  trim_string(indices_str);
  std::stringstream ss(indices_str);
  std::string instr_idx;
  while (std::getline(ss, instr_idx, ' ')) {
    indices.push_back(std::stoi(instr_idx));
  }
  return {kernel_name, indices};
}

void *recv_thread_fun(void *args) {
  CUcontext ctx=(CUcontext)args;
  char *recv_buffer=(char *)malloc(CHANNEL_SIZE);
  while(recv_thread_started) {
    uint32_t bytes=0;
    if(!recv_thread_receiving || !(bytes=channel_host.recv(recv_buffer,CHANNEL_SIZE)))
      continue;
    auto *records=reinterpret_cast<inst_trace_t *>(recv_buffer);
    uint32_t available=bytes/sizeof(inst_trace_t),count=0;
    while(count<available && records[count].cta_id_x!=-1) ++count;
    if(defer_formatting) {
      if(count) {
        FILE *file=ctx_resultsFile[ctx];
        accelsim_trace::write_capture_records(file,records,count);
        for(uint32_t i=0;i<count;++i)
          if(records[i].inst_type==INST_TMA &&
             (records[i].active_mask & records[i].predicate_mask))
            accelsim_trace::write_capture_tma(file,i,
                capture_tma(ctx,&records[i],id_to_opcode_map[records[i].opcode_id]));
      }
    } else {
      for(uint32_t i=0;i<count;++i) {
        const std::string opcode=id_to_opcode_map[records[i].opcode_id];
        ctx_formatters[ctx]->write(&records[i],opcode,capture_tma(ctx,&records[i],opcode));
      }
    }
    if(count<available) {
      if(!defer_formatting && ctx_formatters[ctx]) ctx_formatters[ctx]->reset();
      recv_thread_receiving=false;
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  ctx_current_kernel_name[ctx] = "";
  recv_thread_started = true;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
  pthread_create(&recv_thread, NULL, recv_thread_fun, ctx);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  if (recv_thread_started) {
    recv_thread_started = false;
    pthread_join(recv_thread, NULL);
    if(defer_formatting) {
      for(FILE *file:ctx_deferred_files[ctx]) {
        accelsim_trace::finish_capture(file,id_to_opcode_map);
        if(fclose(file)) throw std::runtime_error("Closing collected GPU records failed");
      }
      ctx_deferred_files.erase(ctx);
    }
  }
}

void nvbit_at_ctx_init(CUcontext ctx) {
  // Everytime we init a context, add the foldername and kernelid to the set
  char buffer[2048];
  sprintf(buffer, "kernelslist_ctx_0x%lx", ctx);
  std::string tmp_kernelslist = user_folder + "/traces/" + buffer;
  ctx_kernelslist[ctx] = tmp_kernelslist;
  sprintf(buffer, "stats_ctx_0x%lx", ctx);
  std::string tmp_stats = user_folder + "/traces/" + buffer;
  ctx_stats_location[ctx] = tmp_stats;
  ctx_kernelid[ctx] = 1;
}
