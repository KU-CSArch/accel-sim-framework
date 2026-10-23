#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>
static void check(cudaError_t status) {
  if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void vector_add(const uint32_t *a,const uint32_t *b,uint32_t *c,uint32_t count) {
  uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count)c[i]=a[i]+b[i];
}
static void write(const char *name,const std::vector<uint32_t> &values) {
  std::ofstream output(name,std::ios::binary);
  output.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(uint32_t));
  if(!output)throw std::runtime_error("output file write failed");
}
int main(int argc,char **argv) {
  try {
    uint32_t count=argc>1?std::strtoul(argv[1],nullptr,10):(1u<<20);
    std::vector<uint32_t>a(count),b(count),c(count),gold(count);
    for(uint32_t i=0;i<count;++i){a[i]=i%1024;b[i]=(3*i)%1024;gold[i]=a[i]+b[i];}
    write("golden.bin",gold);uint32_t *da,*db,*dc;size_t bytes=count*sizeof(uint32_t);
    check(cudaMalloc(&da,bytes));check(cudaMalloc(&db,bytes));check(cudaMalloc(&dc,bytes));
    check(cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice));
    uint64_t errors[4]={};
    for(int launch=0;launch<4;++launch){
      vector_add<<<(count+255)/256,256>>>(da,db,dc,count);check(cudaGetLastError());check(cudaDeviceSynchronize());
      check(cudaMemcpy(c.data(),dc,bytes,cudaMemcpyDeviceToHost));
      for(uint32_t i=0;i<count;++i)errors[launch]+=c[i]!=gold[i];
      char name[64];snprintf(name,sizeof(name),"output-launch-%d.bin",launch);write(name,c);
    }
    check(cudaFree(da));check(cudaFree(db));check(cudaFree(dc));
    printf("{\"experiment\":\"trace-vector-add\",\"elements\":%u,\"blocks\":%u,\"threads_per_block\":256,\"trials\":[",count,(count+255)/256);
    for(int launch=0;launch<4;++launch)printf("%s{\"launch\":%d,\"warmup\":%s,\"mismatches\":%lu}",launch?",":"",launch,launch<3?"true":"false",errors[launch]);
    printf("]}\n");return errors[0]||errors[1]||errors[2]||errors[3];
  }catch(const std::exception &error){fprintf(stderr,"%s\n",error.what());return 1;}
}
