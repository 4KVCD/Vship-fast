// libjxl's AddSupersampled2x (lib/jxl/butteraugli/butteraugli.cc): diffmap = diffmap*(1 - 0.3*w) + w*small, small
// read at half resolution. libjxl multiplies by 1 - 0.3*w in double (its kHeuristicMixingValue is a double), rounds
// to float, then adds w*small in float without fusing (that code is compiled for the baseline CPU).
//
// The double product is computed in double-float (exactMathModule), which gives libjxl's float whenever its rounding
// is decided with a margin far above both computations' errors; otherwise (rarely) it is recomputed in double (FP64)
// or, on GPUs without 64-bit floats, the double-float result is kept, an approximation.

import exactMathModule;

static const float MARGIN = 9.094947e-13f; // 2^-40, relative to the product

[shader("compute")]
[numthreads(16, 16, 1)]
void addsupersample2X(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* diffmap, uniform float* diffmapsmall, uniform uint64_t width, uniform uint64_t height, uniform float w, uniform uint zero){
    const uint64_t x = threadIdx.x + blockIdx.x*16;
    const uint64_t y = threadIdx.y + blockIdx.y*16;
    const uint64_t neww = (width-1)/2 + 1;
    if (x >= width || y >= height) return;

    const uint64_t i = y*width + x;
    const float value = diffmap[i];
    // 1 - 0.3*w, 0.3 being libjxl's double kHeuristicMixingValue as a float pair
    const FF mix = ffAddFloat(ffNeg(ffMulFloat(ff(0.30000001192092896f, -1.1920929132713809e-08f), w, zero)), 1.0f, zero);
    const FF product = ffMulFloat(mix, value, zero);
    float scaled = product.hi;
#if FP64
    if (!decided(product, MARGIN*abs(product.hi))) scaled = float(double(value)*(double(1.0f) - asdouble(0x33333333u, 0x3fd33333u)*double(w)));
#endif
    diffmap[i] = scaled + w*diffmapsmall[(y/2)*neww + x/2];
}
