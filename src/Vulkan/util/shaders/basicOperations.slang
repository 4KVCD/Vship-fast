[shader("compute")]
[numthreads(256, 1, 1)]
void multArray(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform float* src2, uniform uint64_t width){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width) return;
    dst[x] = src[x]*src2[x];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void subArray(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform float* src2, uniform uint64_t width){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width) return;
    dst[x] = src[x]-src2[x];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void memSet(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform uint64_t width, uniform float val){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width) return;
    dst[x] = val;
}