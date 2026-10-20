// libjxl's AddSupersampled2x (lib/jxl/butteraugli/butteraugli.cc): diffmap = diffmap*(1 - 0.3*w) + w*small, small
// read at half resolution. libjxl multiplies by 1 - 0.3*w in double (its kHeuristicMixingValue is a double), rounds
// to float, then adds w*small in float without fusing (that code is compiled for the baseline CPU). The host gives
// that double factor as mixBits; mulDoubleRounded (exactMathModule) computes the product exactly as libjxl does.

import exactMathModule;

[shader("compute")]
[numthreads(16, 16, 1)]
void addsupersample2X(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* diffmap, uniform float* diffmapsmall, uniform uint64_t width, uniform uint64_t height, uniform uint64_t mixBits, uniform float w){
    const uint64_t x = threadIdx.x + blockIdx.x*16;
    const uint64_t y = threadIdx.y + blockIdx.y*16;
    const uint64_t neww = (width-1)/2 + 1;
    if (x >= width || y >= height) return;

    const uint64_t i = y*width + x;
    diffmap[i] = mulDoubleRounded(diffmap[i], mixBits) + w*diffmapsmall[(y/2)*neww + x/2];
}
