#pragma once
#include "trace_formatter.h"
#include <filesystem>

namespace accelsim_trace {
// Native record ABI is recorded explicitly; the CPU reader checks that contract.
struct CaptureMetadata {
  std::string header;
  FormattingOptions options;
  std::map<int, std::string> opcodes;
  uint64_t records_begin = 0;
  uint64_t records_end = 0;
};
void begin_capture(FILE *file, const FormattingOptions &options);
void write_capture_records(FILE *file, const inst_trace_t *records, uint32_t count);
void write_capture_tma(FILE *file, uint32_t index, const TmaSnapshot &snapshot);
void finish_capture(FILE *file, const std::map<int, std::string> &opcodes);
CaptureMetadata read_capture_metadata(FILE *file);
void format_capture(const std::filesystem::path &path);
} // namespace accelsim_trace
