module makeXYBSSIMU2;

// Linear RGB to SSIMULACRA2's positive XYB, with the constants and the arithmetic of libjxl (LinearRGBToXYB in
// lib/jxl/enc_xyb.cc for an intensity target of 255, MakePositiveXYB in tools/ssimulacra2.cc), fma included.

// Returns x. Its bits go through an OR with a zero only known at run time, so the compiler cannot fuse the
// multiplication that produced x into an addition: libjxl rounds these products before adding them.
float unfusedProduct(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

float3 opsin_absorbance(float3 a){
    // libjxl's kOpsinAbsorbanceMatrix (the rows sum to 1 in float) and kOpsinAbsorbanceBias
    const float bias = 0.00379307335f;
    float3 out;
    out.x = fma(0.300000012f, a.x, fma(0.621999979f, a.y, fma(0.0780000016f, a.z, bias)));
    out.y = fma(0.230000004f, a.x, fma(0.691999972f, a.y, fma(0.0780000016f, a.z, bias)));
    out.z = fma(0.243422687f, a.x, fma(0.204767451f, a.y, fma(0.551809847f, a.z, bias)));
    return out;
}

// libjxl's CubeRootAndAdd (lib/jxl/base/fast_math-inl.h): cbrt(x) + add for x >= 0, from an estimate made on the
// exponent bits refined by 4 Newton steps.
float cubeRootAndAdd(float x, float add){
    const float k1_3 = 1.0f/3.0f;
    const float k4_3 = 4.0f/3.0f;
    const float x_3 = k1_3*x;
    const int bits = asint(x);
    // 0 has an exponent of 0, which the estimate below would turn into a NaN
    float r = asfloat((bits == 0) ? 0 : 0x54800000 - (bits >> 23)*0x002AAAAA);
    [ForceUnroll]
    for (int i = 0; i < 3; i++){
        const float r2 = r*r;
        r = fma(-x_3, r2*r2, k4_3*r);
    }
    float r2 = r*r;
    r = fma(k1_3, fma(-x, r2*r2, r), r);
    r2 = r*r;
    return fma(r2, x, add);
}

float3 linear_rgb_to_xyb(float3 a){
    // -cbrtf(bias) as libjxl computes it at run time (ComputePremulAbsorb). The Windows C runtime's cbrtf, which the
    // libjxl tools we compare with use, returns 0x1.3f64ecp-3, one ulp above the correctly rounded cube root, and
    // SSIMULACRA2's coarse scales are sensitive enough to that ulp to move scores by thousandths.
    const float negBiasCbrt = -0.155954212f;
    a = max(opsin_absorbance(a), 0.0f);
    a.x = cubeRootAndAdd(a.x, negBiasCbrt);
    a.y = cubeRootAndAdd(a.y, negBiasCbrt);
    a.z = cubeRootAndAdd(a.z, negBiasCbrt);
    return float3(0.5f*(a.x - a.y), 0.5f*(a.x + a.y), a.z);
}

float3 make_positive_xyb(float3 a, uint zero) {
    a.z = (a.z - a.y) + 0.55f;
    a.x = unfusedProduct(a.x*14.0f, zero) + 0.42f;
    a.y += 0.01f;
    return a;
}

public float3 rgb_to_positive_xyb_d(float3 a, uint zero){
    a = linear_rgb_to_xyb(a);
    a = make_positive_xyb(a, zero);
    return a;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rgb_to_positive_xyb(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p0, uniform float* p1, uniform float* p2, uniform int64_t width, uniform uint zero){
    int64_t x = threadIdx.x + blockIdx.x*256;
    if (x >= width) return;
    float3 packed = {p0[x], p1[x], p2[x]};
    packed = rgb_to_positive_xyb_d(packed, zero);
    p0[x] = packed.x;
    p1[x] = packed.y;
    p2[x] = packed.z;
}
