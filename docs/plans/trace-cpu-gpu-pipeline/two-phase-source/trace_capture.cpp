#include "trace_capture.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace accelsim_trace {
namespace {
constexpr char capture_magic[8] = {'A','C','T','R','A','W','0','1'};
void write_bytes(FILE *file, const void *data, size_t size) {
  if (size && fwrite(data, 1, size, file) != size)
    throw std::runtime_error("Writing collected GPU records failed");
}
void read_bytes(FILE *file, void *data, size_t size) {
  if (size && fread(data, 1, size, file) != size)
    throw std::runtime_error("Collected GPU trace is incomplete");
}
template<class T> void put(FILE *file, T value) { write_bytes(file, &value, sizeof(value)); }
template<class T> T get(FILE *file) { T value{}; read_bytes(file, &value, sizeof(value)); return value; }
void put_string(FILE *file, const std::string &value) {
  put<uint64_t>(file, value.size()); write_bytes(file, value.data(), value.size());
}
std::string get_string(FILE *file) {
  uint64_t size=get<uint64_t>(file); std::string value(size, '\0');
  read_bytes(file, value.data(), size); return value;
}
std::string shell_quote(const std::string &value) {
  std::string quoted="'";
  for (char c:value) quoted += c=='\'' ? "'\\''" : std::string(1,c);
  return quoted+"'";
}
TmaSnapshot get_tma(FILE *file) {
  TmaSnapshot t;
  t.transfer_size=get<uint32_t>(file);
  t.destination_is_shared=get<uint32_t>(file);
  t.mbar_valid=get<uint32_t>(file);
  t.mbar_address=get<uint32_t>(file);
  t.multicast=get<uint32_t>(file);
  t.multicast_mask=get<uint32_t>(file);
  t.byte_count=get<uint64_t>(file);
  t.oob_byte_count=get<uint64_t>(file);
  t.transfer_count=get<uint32_t>(file);
  uint64_t count=get<uint64_t>(file);t.global_addresses.resize(count);
  read_bytes(file,t.global_addresses.data(),count*sizeof(uint64_t));
  return t;
}
} // namespace

void begin_capture(FILE *file, const FormattingOptions &o) {
  put<uint8_t>(file,0); // end the preserved original kernel header
  write_bytes(file,capture_magic,sizeof(capture_magic));
  put<uint32_t>(file,0x01020304); // native endianness
  put<uint32_t>(file,sizeof(inst_trace_t));
  put<uint32_t>(file,alignof(inst_trace_t));
  put<uint32_t>(file,offsetof(inst_trace_t,inst));
  for (int value:{o.address_compression,o.print_core_id,o.lineinfo,
                 o.register_values,o.raw_xz_compression,o.spinlock_mode,
                 o.spinlock_iterations,int(o.spinlock_metadata_present)})
    put<int32_t>(file,value);
  put<uint64_t>(file,o.spinlock_indices.size());
  write_bytes(file,o.spinlock_indices.data(),o.spinlock_indices.size()*sizeof(uint32_t));
}
void write_capture_records(FILE *file,const inst_trace_t *records,uint32_t count) {
  // One bulk copy per received NVBit chunk; no text formatting or compression.
  put<uint32_t>(file,count);
  write_bytes(file,records,size_t(count)*sizeof(inst_trace_t));
  uint32_t tmas=0;
  for (uint32_t i=0;i<count;++i)
    if (records[i].inst_type==INST_TMA &&
        (records[i].active_mask & records[i].predicate_mask)) ++tmas;
  put<uint32_t>(file,tmas);
}
void write_capture_tma(FILE *file,uint32_t index,const TmaSnapshot &t) {
  put<uint32_t>(file,index);
  for (uint32_t value:{t.transfer_size,t.destination_is_shared,t.mbar_valid,
                      t.mbar_address,t.multicast,t.multicast_mask})
    put<uint32_t>(file,value);
  put<uint64_t>(file,t.byte_count);put<uint64_t>(file,t.oob_byte_count);
  put<uint32_t>(file,t.transfer_count);
  put<uint64_t>(file,t.global_addresses.size());
  write_bytes(file,t.global_addresses.data(),t.global_addresses.size()*sizeof(uint64_t));
}
void finish_capture(FILE *file,const std::map<int,std::string> &opcodes) {
  uint64_t footer=ftello(file);
  put<uint64_t>(file,opcodes.size());
  for (const auto &item:opcodes) {
    put<int32_t>(file,item.first);put_string(file,item.second);
  }
  put<uint64_t>(file,footer);
}
CaptureMetadata read_capture_metadata(FILE *file) {
  CaptureMetadata m;
  int c;
  while ((c=fgetc(file))!=EOF && c!=0) m.header.push_back(char(c));
  if (c!=0) throw std::runtime_error("Missing collected-trace header");
  char magic[8];read_bytes(file,magic,sizeof(magic));
  if (memcmp(magic,capture_magic,sizeof(magic)) ||
      get<uint32_t>(file)!=0x01020304 ||
      get<uint32_t>(file)!=sizeof(inst_trace_t) ||
      get<uint32_t>(file)!=alignof(inst_trace_t) ||
      get<uint32_t>(file)!=offsetof(inst_trace_t,inst))
    throw std::runtime_error("Collected trace ABI differs from this CPU reader");
  auto &o=m.options;
  o.address_compression=get<int32_t>(file);o.print_core_id=get<int32_t>(file);
  o.lineinfo=get<int32_t>(file);o.register_values=get<int32_t>(file);
  o.raw_xz_compression=get<int32_t>(file);o.spinlock_mode=get<int32_t>(file);
  o.spinlock_iterations=get<int32_t>(file);o.spinlock_metadata_present=get<int32_t>(file);
  uint64_t n=get<uint64_t>(file);o.spinlock_indices.resize(n);
  read_bytes(file,o.spinlock_indices.data(),n*sizeof(uint32_t));
  m.records_begin=ftello(file);
  if (fseeko(file,-off_t(sizeof(uint64_t)),SEEK_END))
    throw std::runtime_error("Missing collected-trace footer");
  uint64_t footer_end=ftello(file);m.records_end=get<uint64_t>(file);
  if (m.records_end<m.records_begin || m.records_end>footer_end ||
      fseeko(file,m.records_end,SEEK_SET))
    throw std::runtime_error("Invalid collected-trace footer");
  n=get<uint64_t>(file);
  for(uint64_t i=0;i<n;++i) {
    int32_t id=get<int32_t>(file);m.opcodes.emplace(id,get_string(file));
  }
  if (uint64_t(ftello(file))!=footer_end)
    throw std::runtime_error("Collected-trace metadata length differs");
  if (fseeko(file,m.records_begin,SEEK_SET))
    throw std::runtime_error("Cannot seek to GPU records");
  return m;
}
void format_capture(const std::filesystem::path &path) {
  FILE *file=fopen(path.c_str(),"rb");
  if (!file) throw std::runtime_error("Cannot open "+path.string());
  FILE *output=nullptr;bool compressed=false;
  try {
    auto metadata=read_capture_metadata(file);
    std::string destination=path.string();
    if (destination.size()<8 || destination.substr(destination.size()-8)!=".records")
      throw std::runtime_error("Expected a .records input");
    destination.resize(destination.size()-8);
    compressed=metadata.options.raw_xz_compression;
    if (compressed) {
      destination+=".xz";
      std::string command="xz -1 -T0 > "+shell_quote(destination);
      output=popen(command.c_str(),"w");
    } else output=fopen(destination.c_str(),"w");
    if(!output) throw std::runtime_error("Cannot create "+destination);
    write_bytes(output,metadata.header.data(),metadata.header.size());
    TraceFormatter formatter(output,metadata.options);
    const TmaSnapshot empty;
    while(uint64_t(ftello(file))<metadata.records_end) {
      uint32_t count=get<uint32_t>(file);
      if(size_t(count)*sizeof(inst_trace_t)>metadata.records_end-uint64_t(ftello(file)))
        throw std::runtime_error("Incomplete GPU record block");
      std::vector<inst_trace_t> records(count);
      read_bytes(file,records.data(),size_t(count)*sizeof(inst_trace_t));
      uint32_t tmas=get<uint32_t>(file);uint32_t cursor=0;
      for(uint32_t k=0;k<tmas;++k) {
        uint32_t index=get<uint32_t>(file);
        if(index<cursor || index>=count) throw std::runtime_error("Invalid TMA record index");
        while(cursor<index) {
          auto &record=records[cursor++];
          formatter.write(&record,metadata.opcodes.at(record.opcode_id),empty);
        }
        auto snapshot=get_tma(file);auto &record=records[cursor++];
        formatter.write(&record,metadata.opcodes.at(record.opcode_id),snapshot);
      }
      while(cursor<count) {
        auto &record=records[cursor++];
        formatter.write(&record,metadata.opcodes.at(record.opcode_id),empty);
      }
    }
    if(uint64_t(ftello(file))!=metadata.records_end)
      throw std::runtime_error("GPU records overlap their metadata");
    bool failed=ferror(output);
    int status=compressed?pclose(output):fclose(output);output=nullptr;
    if(status || failed) throw std::runtime_error("Writing CPU trace output failed");
    fclose(file);file=nullptr;
  } catch(...) {
    if(output) {if(compressed)pclose(output);else fclose(output);}
    if(file)fclose(file);
    throw;
  }
}
} // namespace accelsim_trace
