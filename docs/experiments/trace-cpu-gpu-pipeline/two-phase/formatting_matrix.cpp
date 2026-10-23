#include "/work/util/tracer_nvbit/tracer_tool/formatting/trace_capture.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
using namespace accelsim_trace;
int main(int argc,char **argv) {
  assert(argc==3);FILE *input=fopen(argv[1],"rb");assert(input);
  auto metadata=read_capture_metadata(input);std::vector<inst_trace_t> records;
  while(uint64_t(ftello(input))<metadata.records_end) {
    uint32_t count;assert(fread(&count,4,1,input)==1);
    std::vector<inst_trace_t> block(count);assert(fread(block.data(),sizeof(inst_trace_t),count,input)==count);
    uint32_t tmas;assert(fread(&tmas,4,1,input)==1 && tmas==0);
    records.insert(records.end(),block.begin(),block.end());
  }
  fclose(input);assert(records.size()>2048);
  records.erase(records.begin()+1024,records.end()-1024);
  std::set<uint32_t> selected;
  for(size_t i=16;i<records.size();i+=32)selected.insert(records[i].instr_idx);
  std::filesystem::path root(argv[2]);std::filesystem::create_directories(root);
  for(int mode=0;mode<3;++mode)for(int flags=0;flags<16;++flags) {
    auto options=metadata.options;options.raw_xz_compression=0;
    options.address_compression=flags&1;options.print_core_id=(flags>>1)&1;
    options.lineinfo=(flags>>2)&1;options.register_values=(flags>>3)&1;
    options.spinlock_mode=mode;options.spinlock_iterations=1;
    options.spinlock_metadata_present=true;
    options.spinlock_indices.assign(selected.begin(),selected.end());
    auto raw=root/("mode"+std::to_string(mode)+"-flags"+std::to_string(flags)+".trace.records");
    FILE *output=fopen(raw.c_str(),"wb");assert(output);
    fwrite(metadata.header.data(),1,metadata.header.size(),output);
    begin_capture(output,options);write_capture_records(output,records.data(),records.size());
    finish_capture(output,metadata.opcodes);assert(fclose(output)==0);
    FILE *check=fopen(raw.c_str(),"rb");assert(check);auto read=read_capture_metadata(check);
    uint32_t count;assert(fread(&count,4,1,check)==1 && count==records.size());
    std::vector<inst_trace_t> restored(count);assert(fread(restored.data(),sizeof(inst_trace_t),count,check)==count);
    assert(memcmp(restored.data(),records.data(),count*sizeof(inst_trace_t))==0);
    uint32_t tmas;assert(fread(&tmas,4,1,check)==1 && !tmas);assert(uint64_t(ftello(check))==read.records_end);fclose(check);
    format_capture(raw);std::cout<<raw.string()<<"\n";
  }
}
