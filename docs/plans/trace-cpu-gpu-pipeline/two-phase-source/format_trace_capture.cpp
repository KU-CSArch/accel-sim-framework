#include "trace_capture.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
int main(int argc,char **argv) {
  if(argc!=2) {std::cerr<<"Usage: format-trace-capture <trace-directory-or-record-file>\n";return 2;}
  try {
    std::filesystem::path input(argv[1]);std::vector<std::filesystem::path> files;
    if(std::filesystem::is_directory(input)) {
      for(const auto &entry:std::filesystem::directory_iterator(input))
        if(entry.path().extension()==".records") files.push_back(entry.path());
    } else files.push_back(input);
    std::sort(files.begin(),files.end());
    if(files.empty()) throw std::runtime_error("No collected GPU record files");
    for(const auto &file:files) {
      accelsim_trace::format_capture(file);
      std::cout<<"Formatted "<<file.string()<<"\n";
    }
  } catch(const std::exception &error) {std::cerr<<error.what()<<"\n";return 1;}
}
