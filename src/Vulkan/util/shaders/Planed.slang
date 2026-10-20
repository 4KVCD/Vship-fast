[shader("compute")]
[numthreads(256, 1, 1)]
void StrideEliminator(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t stride, uniform uint64_t width, uniform uint64_t height){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width*height) return;
    uint64_t j = x%width;
    uint64_t i = x/width;
    // the byte offset added to the address as an integer: SPIRV-Cross (MoltenVK) mistranslates byte-pointer arithmetic
    dst[i*width+j] = ((float*)((uint64_t)src + i*stride))[j];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void StrideAdder(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t stride, uniform uint64_t width, uniform uint64_t height){
    uint64_t x = threadIdx.x + 256*blockIdx.x;
    if (x >= width*height) return;
    uint64_t j = x%width;
    uint64_t i = x/width;
    // the byte offset added to the address as an integer: SPIRV-Cross (MoltenVK) mistranslates byte-pointer arithmetic
    ((float*)((uint64_t)src + i*stride))[j] = dst[i*width+j];
}