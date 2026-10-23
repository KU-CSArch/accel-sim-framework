#include <cassert>
#include <array>
#include <map>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zstd.h>
#include "/source/util/tracer_nvbit/tracer_tool/tracez_format.h"

int main(int argc, char **argv) {
  assert(argc == 3);
  std::ifstream raw(argv[1]);
  assert(raw.is_open());
  using Key = std::array<unsigned,11>;
  std::map<Key,std::vector<std::string>> expected_by_warp;
  std::string line;
  while (std::getline(raw, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '-') continue;
    std::istringstream row(line);
    Key fields{};
    for (auto &field : fields) row >> field;
    assert(row);
    std::getline(row, line);
    assert(!line.empty() && line[0] == ' ');
    expected_by_warp[fields].push_back(line.substr(1));
  }
  FILE *fp = fopen(argv[2], "rb");
  assert(fp);
  auto header = tracez::read_kernel_header(fp);
  uint32_t dict_size, subchunk_insts;
  tracez::read_compression_header(fp, dict_size, subchunk_insts);
  std::vector<char> dict(dict_size);
  assert(fread(dict.data(), 1, dict_size, fp) == dict_size);
  std::vector<tracez::tb_index_t> index;
  tracez::read_index(fp, index);
  size_t total=0,warps=0;
  for(const auto &tb:index)for(size_t w=0;w<tb.warps.size();++w) {
  Key key{tb.tb_id_x,tb.tb_id_y,tb.tb_id_z,unsigned(w),tb.cluster_id_x,
          tb.cluster_id_y,tb.cluster_id_z,tb.cluster_cta_id_x,tb.cluster_cta_id_y,
          tb.cluster_cta_id_z,tb.cluster_rank};
  const auto &expected=expected_by_warp.at(key);
  const auto &warp=tb.warps[w];
  assert(warp.inst_count==expected.size());
  assert(fseeko(fp,warp.file_offset,SEEK_SET)==0);
  uint32_t count,chunks;
  assert(fread(&count,4,1,fp)==1);
  assert(fread(&chunks,4,1,fp)==1);
  assert(count==expected.size() && chunks==warp.num_subchunks);
  auto *context = ZSTD_createDCtx();
  auto *dictionary = dict_size ? ZSTD_createDDict(dict.data(), dict_size) : nullptr;
  size_t matched = 0;
  for (uint32_t chunk = 0; chunk < chunks; ++chunk) {
    uint32_t compressed_size;
    assert(fread(&compressed_size, 4, 1, fp) == 1);
    std::vector<char> compressed(compressed_size);
    assert(fread(compressed.data(), 1, compressed_size, fp) == compressed_size);
    auto decoded_size = ZSTD_getFrameContentSize(compressed.data(), compressed_size);
    assert(decoded_size != ZSTD_CONTENTSIZE_UNKNOWN && decoded_size != ZSTD_CONTENTSIZE_ERROR);
    std::string text(decoded_size, '\0');
    size_t decoded = dictionary
      ? ZSTD_decompress_usingDDict(context, text.data(), text.size(), compressed.data(), compressed_size, dictionary)
      : ZSTD_decompressDCtx(context, text.data(), text.size(), compressed.data(), compressed_size);
    assert(!ZSTD_isError(decoded) && decoded == text.size());
    std::istringstream rows(text);
    size_t chunk_count = 0;
    while (std::getline(rows, line)) {
      assert(matched < expected.size());
      if (line != expected[matched]) {
        std::cerr << "instruction record differs at " << matched << "\n";
        return 2;
      }
      ++matched; ++chunk_count;
    }
    assert(chunk_count == warp.subchunk_inst_counts[chunk]);
  }
  assert(matched == expected.size());
  ZSTD_freeDDict(dictionary); ZSTD_freeDCtx(context);
  total+=matched;++warps;
  }
  assert(warps==expected_by_warp.size());fclose(fp);
  std::cout << total << "\n";
}
