// Test-only reference: original Accel-Sim v2.0.0 formatting body.
#include <nvbit.h>
#include "/work/util/tracer_nvbit/tracer_tool/formatting/trace_capture.h"
#include <algorithm>
#include <cassert>
#include <bitset>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
using namespace accelsim_trace;
enum address_format {list_all=0,base_stride=1,base_delta=2,tma_list_all=3,tma_base_delta=4};
const int SPINLOCK_MODE_NONE=0,SPINLOCK_MODE_FAST_FORWARD=1,SPINLOCK_MODE_MARK_REGION=2;
int allow_reg_val_tracing,enable_compress,print_core_id,lineinfo,spinlock_handling_mode,spinlock_iter_to_keep;
std::map<CUcontext,FILE*> ctx_resultsFile;
std::map<int,std::string> id_to_opcode_map;
std::map<CUcontext,std::string> ctx_current_kernel_name;
std::map<std::string,std::vector<uint32_t>*> spinlock_instr_map;
TMATransferInfo_t nvbit_parse_tma_transfer_info(CUcontext,const char*,uint8_t*,size_t) {
  throw std::runtime_error("TMA is target-missing in this A6000 reference test");
}
void nvbit_parse_tma_src_addrs(CUcontext,const char*,uint8_t*,size_t,nvbit_tma_src_addrs_callback_t,void*) { throw std::runtime_error("TMA target missing"); }
void nvbit_parse_tma_dst_addrs(CUcontext,const char*,uint8_t*,size_t,nvbit_tma_dst_addrs_callback_t,void*) { throw std::runtime_error("TMA target missing"); }
bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {
  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        if (stride != addrs[s] - addrs[s - 1]) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {
  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

void base_delta_compress_tma(const uint64_t *addrs, const size_t num_addrs,
                             const std::bitset<32> &mask, uint64_t &base_addr,
                             std::vector<long long> &deltas) {
  // TMA version for delta compression
  bool warp_active = mask.any() && num_addrs > 1;
  if (warp_active) {
    base_addr = addrs[0];
    for (size_t i = 1; i < num_addrs; i++) {
      deltas.push_back(addrs[i] - addrs[i - 1]);
    }
  }
}

typedef std::map<uint32_t, size_t> counter_t;
typedef std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> warp_key_t;

counter_t create_counter(const std::vector<uint32_t> &indices) {
  counter_t counter;
  for (auto instr_idx : indices) {
    counter[instr_idx] = 0;
  }
  return counter;
}

/* Emit a REPLAY_START or REPLAY_END marker to the trace file */
void emit_replay_marker(FILE *outfile, inst_trace_t *trace,
                        const char *marker_opcode) {
  // CTA ID
  fprintf(outfile, "%d ", trace->cta_id_x);
  fprintf(outfile, "%d ", trace->cta_id_y);
  fprintf(outfile, "%d ", trace->cta_id_z);
  fprintf(outfile, "%d ", trace->warpid_tb);
  // Cluster info
  fprintf(outfile, "%d ", trace->cluster_id_x);
  fprintf(outfile, "%d ", trace->cluster_id_y);
  fprintf(outfile, "%d ", trace->cluster_id_z);
  fprintf(outfile, "%d ", trace->cluster_cta_id_x);
  fprintf(outfile, "%d ", trace->cluster_cta_id_y);
  fprintf(outfile, "%d ", trace->cluster_cta_id_z);
  fprintf(outfile, "%d ", trace->cluster_rank);
  // PC (use 0000 for markers)
  fprintf(outfile, "0000 ");
  // Mask (all active)
  fprintf(outfile, "ffffffff ");
  // dest_num = 0
  fprintf(outfile, "0 ");
  // Opcode
  fprintf(outfile, "%s ", marker_opcode);
  // src_num = 0, mem_width = 0, imm1 = 0, imm2 = 0
  fprintf(outfile, "0 0 0 0\n");
}

/* Per-warp state for mark_region mode */
struct warp_replay_state_t {
  bool in_region;     // currently inside spinlock region
  bool start_emitted; // REPLAY_START already written for this region
  int iter_count;     // iterations recorded so far
};


std::map<warp_key_t,counter_t> warp_counter_map;
std::map<warp_key_t,warp_replay_state_t> warp_replay_map;
void reference_write(CUcontext ctx,inst_trace_t *trace) {
        std::string opcode = id_to_opcode_map[trace->opcode_id];
        // only dump reg val if opcode contains: BAR
        bool dump_reg_val = false;
        if (allow_reg_val_tracing &&
            ((opcode.find("BAR") != std::string::npos ||
              opcode.find("HGMMA") != std::string::npos ||
              opcode.find("SYNCS.ARRIVE") != std::string::npos ||
              opcode.find("SYNCS.EXCH.64") != std::string::npos ||
              opcode.find("SYNCS.PHASECHK.TRANS64.TRYWAIT") !=
                  std::string::npos ||
              opcode.find("SYNCS.PHASECHK.TRANS64") != std::string::npos))) {
          // SYNCS: Mbarrier related instructions
          // SYNCS.ARRIVE: equivalent to mbarrier.arrive, register values are
          // the arrival count or expect tx count SYNCS.EXCH.64: equivalent to
          // mbarrier.init, register values are the mbarrier arrival count
          // SYNCS.PHASECHK.TRANS64.TRYWAIT: equivalent to mbarrier.try_wait,
          // register values are the phase this wait is for
          // SYNCS.PHASECHK.TRANS64: equivalent to mbarrier.test_wait,
          // register values are the phase this wait is for
          dump_reg_val = true;
        }

        /* Spinlock handling (fast_forward or mark_region) */
        if (spinlock_handling_mode > SPINLOCK_MODE_NONE) {
          // Check if this warp is in the warp_counter_map
          warp_key_t warp_key =
              std::make_tuple(trace->cta_id_x, trace->cta_id_y, trace->cta_id_z,
                              trace->warpid_tb);
          if (warp_counter_map.find(warp_key) == warp_counter_map.end()) {
            // This warp is not in the warp_counter_map, so we create a counter
            // for this warp using the spinlock instruction indices for the
            // current kernel
            std::vector<uint32_t> &indices =
                *(spinlock_instr_map[ctx_current_kernel_name[ctx]]);
            warp_counter_map[warp_key] = create_counter(indices);
            if (spinlock_handling_mode == SPINLOCK_MODE_MARK_REGION) {
              warp_replay_map[warp_key] = {false, false, 0};
            }
          }

          // Get the counter map for this warp
          auto &counter = warp_counter_map[warp_key];
          bool is_spinlock_instr =
              (counter.find(trace->instr_idx) != counter.end());

          if (spinlock_handling_mode == SPINLOCK_MODE_FAST_FORWARD) {
            // Fast-forward mode: skip instructions beyond threshold
            if (is_spinlock_instr) {
              counter[trace->instr_idx]++;
              if (counter[trace->instr_idx] > spinlock_iter_to_keep) {
                // This spinlock instruction is executed more than the threshold
                // so we fast forward it in the output trace
                // Note we are only fast forwarding the innermost spinlock loop
                return;
              }
            } else {
              // We are exiting the innermost spinlock loop, so we reset the
              // counter map for this warp
              for (auto &[instr_idx, count] : counter) {
                count = 0;
              }
            }
          } else if (spinlock_handling_mode == SPINLOCK_MODE_MARK_REGION) {
            // Mark-region mode: emit REPLAY_START/REPLAY_END markers
            auto &replay_state = warp_replay_map[warp_key];

            if (is_spinlock_instr) {
              counter[trace->instr_idx]++;

              // Entering spinlock region
              if (!replay_state.in_region) {
                replay_state.in_region = true;
                replay_state.start_emitted = false;
                replay_state.iter_count = 1;
              }

              // Emit REPLAY_START before first instruction in region
              if (!replay_state.start_emitted) {
                emit_replay_marker(ctx_resultsFile[ctx], trace, "REPLAY_START");
                replay_state.start_emitted = true;
              }

              // Skip if beyond first iteration
              if (counter[trace->instr_idx] > spinlock_iter_to_keep) {
                return;
              }
              // else: fall through to emit the instruction
            } else {
              // Non-spinlock instruction
              if (replay_state.in_region) {
                // Exiting spinlock region - emit REPLAY_END
                emit_replay_marker(ctx_resultsFile[ctx], trace, "REPLAY_END");
                replay_state.in_region = false;
                replay_state.start_emitted = false;
                // Reset counters
                for (auto &[instr_idx, count] : counter) {
                  count = 0;
                }
              }
              // else: fall through to emit the instruction
            }
          }
        }

        // Dump trace in text
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cta_id_x);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cta_id_y);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cta_id_z);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->warpid_tb);
        // Cluster information
        // Cluster id within the grid
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_id_x);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_id_y);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_id_z);
        // CTA id within the cluster
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_cta_id_x);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_cta_id_y);
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_cta_id_z);
        // CTA rank within the cluster
        fprintf(ctx_resultsFile[ctx], "%d ", trace->cluster_rank);
        if (print_core_id) {
          fprintf(ctx_resultsFile[ctx], "%d ", trace->sm_id);
          fprintf(ctx_resultsFile[ctx], "%d ", trace->warpid_sm);
        }
        if (lineinfo) {
          fprintf(ctx_resultsFile[ctx], "%d ", trace->line_num);
        }
        fprintf(ctx_resultsFile[ctx], "%04x ",
                trace->vpc); // Print the virtual PC
        fprintf(ctx_resultsFile[ctx], "%08x ",
                trace->active_mask & trace->predicate_mask);

        // Helper function to print the register number
        auto print_reg = [&](int reg_num) {
          if (reg_num >= 256) {
            fprintf(ctx_resultsFile[ctx], "UR%d ", reg_num - UREG_OFFSET);
          } else {
            fprintf(ctx_resultsFile[ctx], "R%d ", reg_num);
          }
        };

        // Only regular instructions have register information
        // While TMA instruction also does, but they are all
        // uniform register and we cannot obtain them from
        // iterating instruction operands during instrumentation
        // In future we might want to add those by parsing
        // TMA instruction SASS string for uniform register
        // dependency tracking

        // Dump destination register information
        if (trace->inst_type == TracerInstrType::INST_TMA) {
          // No destination register for TMA instructions
          fprintf(ctx_resultsFile[ctx], "0 ");
        } else {
          if (trace->inst.regular.GPRDst >= 0) { // GPR dst
            fprintf(ctx_resultsFile[ctx], "1 ");
            print_reg(trace->inst.regular.GPRDst);
          } else
            fprintf(ctx_resultsFile[ctx], "0 ");
        }

        // Print the opcode.
        fprintf(ctx_resultsFile[ctx], "%s ",
                id_to_opcode_map[trace->opcode_id].c_str());

        // Dump source registers information
        if (trace->inst_type == TracerInstrType::INST_TMA) {
          // No source registers for TMA instructions
          fprintf(ctx_resultsFile[ctx], "0 ");
        } else {
          // Handle source registers for regular instructions
          unsigned src_count = 0;
          for (int s = 0; s < MAX_SRC; s++) // GPR srcs count.
            if (trace->inst.regular.GPRSrcs[s] >= 0)
              src_count++;
          fprintf(ctx_resultsFile[ctx], "%d ", src_count);

          for (int s = 0; s < MAX_SRC; s++) { // GPR srcs.
            if (trace->inst.regular.GPRSrcs[s] >= 0) {
              print_reg(trace->inst.regular.GPRSrcs[s]);
            }
          }
        }

        // Print is_gmma_commit_group flag
        bool is_gmma_instruction =
            strstr(id_to_opcode_map[trace->opcode_id].c_str(), "GMMA") != NULL;
        if (is_gmma_instruction) {
          fprintf(ctx_resultsFile[ctx], "%d ", trace->is_gmma_commit_group);
        }

        // print addresses
        std::bitset<32> mask(trace->active_mask & trace->predicate_mask);
        if (trace->inst_type == TracerInstrType::INST_REGULAR &&
            trace->inst.regular.is_mem) {
          std::istringstream iss(id_to_opcode_map[trace->opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          while (std::getline(iss, token, '.')) {
            if (!token.empty())
              tokens.push_back(token);
          }
          fprintf(ctx_resultsFile[ctx], "%d ",
                  get_datawidth_from_opcode(tokens));

          bool base_stride_success = false;
          uint64_t base_addr = 0;
          int stride = 0;
          std::vector<long long> deltas;

          if (enable_compress) {
            // try base+stride format
            base_stride_success = base_stride_compress(
                trace->inst.regular.addrs, mask, base_addr, stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_compress(trace->inst.regular.addrs, mask, base_addr,
                                  deltas);
            }
          }

          if (base_stride_success && enable_compress) {
            // base + stride format
            fprintf(ctx_resultsFile[ctx], "%u 0x%lx %d ",
                    address_format::base_stride, base_addr, stride);
          } else if (!base_stride_success && enable_compress) {
            // base + delta format
            fprintf(ctx_resultsFile[ctx], "%u 0x%lx ",
                    address_format::base_delta, base_addr);
            for (int s = 0; s < deltas.size(); s++) {
              fprintf(ctx_resultsFile[ctx], "%lld ", deltas[s]);
            }
          } else {
            // list all the addresses
            fprintf(ctx_resultsFile[ctx], "%u ", address_format::list_all);
            for (int s = 0; s < 32; s++) {
              if (mask.test(s))
                fprintf(ctx_resultsFile[ctx], "0x%016lx ",
                        trace->inst.regular.addrs[s]);
            }
          }
        } else if (trace->inst_type == TracerInstrType::INST_TMA) {
          // TMA instructions
          // Check if the bitmask is all 0, if so, dont parse the TMA
          // instruction
          if ((trace->active_mask & trace->predicate_mask) == 0) {
            // Dummy values to NOP TMA instruction
            // transfer_size mbar_addr is_multicast byte_count oob_byte_count
            // tma_base_delta base_addr transfer_count
            fprintf(ctx_resultsFile[ctx], "0 0x0 0 0 0 4 0x0 0 ");
          } else {
            // Parse the TMA instruction
            const char *opcode_str = id_to_opcode_map[trace->opcode_id].c_str();
            TMATransferInfo_t info = nvbit_parse_tma_transfer_info(
                ctx, opcode_str, trace->inst.tma.tma_param_handle,
                trace->inst.tma.tma_param_handle_size);
            // Get TMA transfer size, which is the data width
            fprintf(ctx_resultsFile[ctx], "%d ", info.transfer_size);

            // Get TMA mbar address
            if (info.dst_memspace ==
                InstrType::MemorySpace::DISTRIBUTED_SHARED) {
              assert(info.dst.shared.is_mbar_valid &&
                     "Invalid TMA mbar address");
              fprintf(ctx_resultsFile[ctx], "0x%08x ",
                      info.dst.shared.mbar_address);
              if (info.is_multicast) {
                // Using multicast
                fprintf(ctx_resultsFile[ctx], "1 ");
                // multicast flags
                fprintf(ctx_resultsFile[ctx], "0x%04x ",
                        info.multicast_cta_mask);
              } else {
                // Not using multicast
                fprintf(ctx_resultsFile[ctx], "0 ");
              }
            } else {
              // mbarrier address, set to 0
              fprintf(ctx_resultsFile[ctx], "0x%08x ", 0);
              // Not using multicast
              fprintf(ctx_resultsFile[ctx], "0 ");
            }
            // This is the actual transfer byte count
            fprintf(ctx_resultsFile[ctx], "%ld ", info.byte_count);

            // This is the oob transfer byte count, if using tensor copy
            if (info.is_tensor) {
              fprintf(ctx_resultsFile[ctx], "%ld ",
                      info.tensor.oob_transfer_count * info.transfer_size);
            } else {
              fprintf(ctx_resultsFile[ctx], "%ld ", 0);
            }

            // Determine the global address using callback API
            // Callback collects inbound (non-OOB) addresses
            struct TMAAddrCollector {
              std::vector<uint64_t> addrs;
              static void callback(const TMAElementAddress_t *raw_addrs,
                                   size_t count, void *user_data) {
                auto *self = static_cast<TMAAddrCollector *>(user_data);
                for (size_t i = 0; i < count; i++) {
                  if (!raw_addrs[i].is_oob) {
                    self->addrs.push_back(raw_addrs[i].address);
                  }
                }
              }
            };
            TMAAddrCollector collector;

            if (info.src_memspace == InstrType::MemorySpace::GLOBAL) {
              nvbit_parse_tma_src_addrs(ctx, opcode_str,
                                        trace->inst.tma.tma_param_handle,
                                        trace->inst.tma.tma_param_handle_size,
                                        TMAAddrCollector::callback, &collector);
            } else if (info.dst_memspace == InstrType::MemorySpace::GLOBAL) {
              nvbit_parse_tma_dst_addrs(ctx, opcode_str,
                                        trace->inst.tma.tma_param_handle,
                                        trace->inst.tma.tma_param_handle_size,
                                        TMAAddrCollector::callback, &collector);
            }

            size_t global_count_inbound = collector.addrs.size();
            uint64_t *global_addrs = collector.addrs.data();

            // Try to compress memory addresses
            uint64_t base_addr = 0;
            std::vector<long long> deltas;
            if (enable_compress) {
              base_delta_compress_tma(global_addrs, global_count_inbound, mask,
                                      base_addr, deltas);
            }

            if (enable_compress) {
              fprintf(ctx_resultsFile[ctx], "%u 0x%lx ",
                      address_format::tma_base_delta, base_addr);
              fprintf(ctx_resultsFile[ctx], "%d ", info.transfer_count);
              for (size_t s = 0; s < deltas.size(); s++) {
                fprintf(ctx_resultsFile[ctx], "%lld ", deltas[s]);
              }
            } else {
              // list all the addresses
              fprintf(ctx_resultsFile[ctx], "%u ",
                      address_format::tma_list_all);
              fprintf(ctx_resultsFile[ctx], "%d ", info.transfer_count);
              for (size_t s = 0; s < global_count_inbound; s++) {
                fprintf(ctx_resultsFile[ctx], "0x%016lx ", global_addrs[s]);
              }
            }
          }
        } else {
          fprintf(ctx_resultsFile[ctx], "0 ");
        }

        // Print the immediate values
        fprintf(ctx_resultsFile[ctx], "%ld ", trace->inst.regular.imm);
        fprintf(ctx_resultsFile[ctx], "%ld ", trace->inst.regular.imm2);

        if (allow_reg_val_tracing) {
          // Trace version will be 6, a Val|NoVal will follow the immediate
          // number Print the register values if dump_reg_val is true
          if (dump_reg_val) {
            fprintf(ctx_resultsFile[ctx], "Val ");

            // dump destination register values if GPRDst >= 0
            if (trace->inst.regular.GPRDst >= 0) {
              auto minmax_pair =
                  std::minmax_element(std::begin(trace->inst.regular.desRegVal),
                                      std::end(trace->inst.regular.desRegVal));
              bool is_all_same = (*minmax_pair.first == *minmax_pair.second);
              if (!is_all_same) {
                fprintf(ctx_resultsFile[ctx], "32 ");
                for (int tid = 0; tid < 32; tid++) {
                  fprintf(ctx_resultsFile[ctx], "%08x ",
                          trace->inst.regular.desRegVal[tid]);
                }
              } else {
                fprintf(ctx_resultsFile[ctx], "1 ");
                fprintf(ctx_resultsFile[ctx], "%08x ", *minmax_pair.first);
              }
            }
            // Build the per-src hi pointer: nullptr means emit lo as 32-bit;
            // non-null means fuse (hi << 32) | lo into 64-bit. Opcode-specific
            // wiring of auxRegVals lives here and nowhere else below.
            const uint32_t *src_hi[MAX_SRC] = {nullptr};
            if (opcode == "SYNCS.EXCH.64") {
              // SYNCS.EXCH.64's 64-bit src2: srcRegVals[1] lo + auxRegVals hi.
              src_hi[1] = trace->inst.regular.auxRegVals;
            }

            for (int s = 0; s < MAX_SRC; s++) {
              if (trace->inst.regular.GPRSrcs[s] < 0)
                continue;
              const uint32_t *lo = trace->inst.regular.srcRegVals[s];
              const uint32_t *hi = src_hi[s];
              if (hi != nullptr) {
                uint64_t v64[32];
                for (int tid = 0; tid < 32; tid++)
                  v64[tid] = ((uint64_t)hi[tid] << 32) | (uint64_t)lo[tid];
                auto mm = std::minmax_element(std::begin(v64), std::end(v64));
                if (*mm.first != *mm.second) {
                  fprintf(ctx_resultsFile[ctx], "32 ");
                  for (int tid = 0; tid < 32; tid++)
                    fprintf(ctx_resultsFile[ctx], "%016lx ", v64[tid]);
                } else {
                  fprintf(ctx_resultsFile[ctx], "1 %016lx ", *mm.first);
                }
              } else {
                auto mm = std::minmax_element(lo, lo + 32);
                if (*mm.first != *mm.second) {
                  fprintf(ctx_resultsFile[ctx], "32 ");
                  for (int tid = 0; tid < 32; tid++)
                    fprintf(ctx_resultsFile[ctx], "%08x ", lo[tid]);
                } else {
                  fprintf(ctx_resultsFile[ctx], "1 %08x ", *mm.first);
                }
              }
            }
          } else {
            fprintf(ctx_resultsFile[ctx], "NoVal ");
          }
        }
        fprintf(ctx_resultsFile[ctx], "\n");

}
int main(int argc,char **argv) {
  if(argc!=3)return 2;
  FILE *input=fopen(argv[1],"rb"),*output=fopen(argv[2],"w");assert(input&&output);
  auto metadata=read_capture_metadata(input);auto &o=metadata.options;
  allow_reg_val_tracing=o.register_values;enable_compress=o.address_compression;
  print_core_id=o.print_core_id;lineinfo=o.lineinfo;spinlock_handling_mode=o.spinlock_mode;
  spinlock_iter_to_keep=o.spinlock_iterations;ctx_resultsFile[nullptr]=output;
  id_to_opcode_map=metadata.opcodes;ctx_current_kernel_name[nullptr]="captured";
  spinlock_instr_map["captured"]=&o.spinlock_indices;
  fwrite(metadata.header.data(),1,metadata.header.size(),output);
  uint64_t count_total=0;
  while(uint64_t(ftello(input))<metadata.records_end) {
    uint32_t count;assert(fread(&count,4,1,input)==1);
    std::vector<inst_trace_t> records(count);
    assert(fread(records.data(),sizeof(inst_trace_t),count,input)==count);
    uint32_t tmas;assert(fread(&tmas,4,1,input)==1);
    if(tmas)throw std::runtime_error("Hardware TMA target missing");
    for(auto &record:records)reference_write(nullptr,&record);
    count_total+=count;
  }
  assert(uint64_t(ftello(input))==metadata.records_end);
  assert(!ferror(output));assert(fclose(output)==0);fclose(input);
  printf("%lu\n",count_total);
}
