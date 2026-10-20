// libjxl's CombineChannelsToDiffmap (lib/jxl/butteraugli/butteraugli.cc): the distortion map from the mask and the
// accumulated differences. Built twice: with -DFP64=1 for GPUs with 64-bit floats, whose MaskY and MaskDcY give
// libjxl's results exactly, and with -DFP64=0 for the others, which approximate them very closely.
//
// libjxl compiles this function for the CPU it runs on (AVX2 here), and its compiler fuses each multiplication and
// addition written in one C++ expression: (scaler * delta) + offset, and MaskColor's
// color[0] * mask + color[1] * mask + color[2] * mask, whose first sum fuses its left product. Those are the fma()
// calls below; every other operation is rounded on its own.

import exactMathModule;

// libjxl's kGlobalScale, 1 / (17.83f * 0.79079917404f), stored as a float
static const float kGlobalScale = 0.07092224061489105f;

// libjxl's MaskY and MaskDcY: (kGlobalScale*(1 + mul/(scaler*delta + offset)))^2, in double, rounded to float.
// Computed in double-float (exactMathModule), which gives libjxl's float whenever its rounding is decided with a margin far
// above both computations' errors; otherwise (rarely) recomputed in double (FP64) or, on GPUs without 64-bit floats,
// the double-float result is kept, an approximation. The constants are libjxl's doubles: as float pairs for the
// double-float computation, from their bits for the double one.
static const float MARGIN = 9.094947e-13f; // 2^-40, relative to the result

FF maskFF(float delta, FF offset, FF scaler, FF mul, uint zero){
    const FF c = ffDiv(mul, ffAdd(ffMulFloat(scaler, delta, zero), offset, zero), zero);
    const FF retval = ffMulFloat(ffAddFloat(c, 1.0f, zero), kGlobalScale, zero);
    return ffMul(retval, retval, zero);
}

#if FP64
float maskDouble(float delta, double offset, double scaler, double mul){
    const double c = divide64(mul, fma(scaler, double(delta), offset));
    const double retval = double(kGlobalScale)*(double(1.0f) + c);
    return float(retval*retval);
}
#endif

float maskY(float delta, uint zero){
    const FF m = maskFF(delta, ff(0.8295917510986328f, 3.843367402112108e-09f), ff(0.4519369304180145f, -8.215014446477653e-09f), ff(2.5485944747924805f, 4.507519690832851e-09f), zero);
#if FP64
    if (!decided(m, MARGIN*m.hi)) return maskDouble(delta, asdouble(0x02103a76u, 0x3fea8c04u), asdouble(0xd72ddfbfu, 0x3fdcec88u), asdouble(0x809ae090u, 0x40046385u));
#endif
    return m.hi;
}

float maskDcY(float delta, uint zero){
    const FF m = maskFF(delta, ff(0.20025578141212463f, 3.807875348371681e-09f), ff(3.8744940757751465f, 1.1226485696624877e-07f), ff(0.5050545334815979f, -8.462597733682742e-09f), zero);
#if FP64
    if (!decided(m, MARGIN*m.hi)) return maskDouble(delta, asdouble(0x482d66d1u, 0x3fc9a1fbu), asdouble(0xcf116418u, 0x400efef6u), asdouble(0x1b74e8ceu, 0x3fe02968u));
#endif
    return m.hi;
}

// libjxl's MaskColor
float maskColor(float c0, float c1, float c2, float mask){
    return fma(c2, mask, fma(c0, mask, c1*mask));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeDiffmap(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* mask, uniform float* block_diff_dc0, uniform float* block_diff_dc1, uniform float* block_diff_dc2, uniform float* block_diff_ac0, uniform float* block_diff_ac1, uniform float* block_diff_ac2, uniform float* dst, uniform uint64_t width, uniform uint zero){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float val = mask[x];
    const float maskval = maskY(val, zero);
    const float maskval_dc = maskDcY(val, zero);

    // libjxl multiplies the X channel's differences by xmul, 1 by default
    dst[x] = sqrtRounded(maskColor(block_diff_dc0[x], block_diff_dc1[x], block_diff_dc2[x], maskval_dc) + maskColor(block_diff_ac0[x], block_diff_ac1[x], block_diff_ac2[x], maskval));
}
