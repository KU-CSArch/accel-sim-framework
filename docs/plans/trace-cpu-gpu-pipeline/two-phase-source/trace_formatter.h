#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "../trace_record.h"

namespace accelsim_trace {
struct FormattingOptions {
  int address_compression = 1;
  int print_core_id = 0;
  int lineinfo = 0;
  int register_values = 1;
  int raw_xz_compression = 1;
  int spinlock_mode = 0;
  int spinlock_iterations = 1;
  bool spinlock_metadata_present = false;
  std::vector<uint32_t> spinlock_indices;
};

// Values returned by the existing NVBit TMA APIs while their CUDA context exists.
// The CPU formatter has no CUDA dependency and consumes these saved values only.
struct TmaSnapshot {
  uint32_t transfer_size = 0;
  uint32_t destination_is_shared = 0;
  uint32_t mbar_valid = 0;
  uint32_t mbar_address = 0;
  uint32_t multicast = 0;
  uint32_t multicast_mask = 0;
  uint64_t byte_count = 0;
  uint64_t oob_byte_count = 0;
  uint32_t transfer_count = 0;
  std::vector<uint64_t> global_addresses;
};

class TraceFormatter {
 public:
  TraceFormatter(FILE *output, const FormattingOptions &options);
  void write(inst_trace_t *trace, const std::string &opcode,
             const TmaSnapshot &tma);
  void reset();
 private:
  using WarpKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;
  using Counter = std::map<uint32_t, size_t>;
  struct ReplayState {
    bool in_region;
    bool start_emitted;
    int iter_count;
  };
  FILE *output_;
  FormattingOptions options_;
  std::map<WarpKey, Counter> warp_counter_map;
  std::map<WarpKey, ReplayState> warp_replay_map;
};
} // namespace accelsim_trace
