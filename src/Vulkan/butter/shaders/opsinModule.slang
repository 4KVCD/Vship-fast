module opsinModule;

// Butteraugli's opsin dynamics (libjxl's OpsinDynamicsImage), from the image and its blur (sigma 1.2)

import exactMathModule;

// libjxl's FastLog2f (base/fast_math-inl.h): x's exponent and mantissa split around 2/3 with integer operations,
// then a 2,2 rational polynomial of the mantissa - 1 in Horner form (fused multiply-adds), divided, plus the
// exponent. Every operation is libjxl's, so the result is the same float on any GPU, unlike the driver's log().
float fastLog2f(float x){
    const int bits = asint(x);
    const int exponent = (bits - 0x3f2aaaab) >> 23;
    const float m = asfloat(bits - (exponent << 23)) - 1.0f;
    const float yp = fma(fma(7.4245873327820566E-01f, m, 1.4287160470083755E+00f), m, -1.8503833400518310E-06f);
    const float yq = fma(fma(1.7409343003366853E-01f, m, 1.0096718572241148E+00f), m, 9.9032814277590719E-01f);
    return divide(yp, yq) + float(exponent);
}

// libjxl's Gamma: 19.245013259874995 * ln(v + 9.9710635769299145) - 23.16046239805755, with the log as
// FastLog2f times ln(2). 13.339627265930176f is libjxl's 19.245013259874995f * kInvLog2e, a product of two floats.
public float gamma(float v) {
    v = max(v, 0.0f);
    return fma(13.339627265930176f, fastLog2f(v + 9.9710635769299145f), -23.16046239805755f);
}

public float3 butterOpsinAbsorbance(float3 a, bool clamp = false){
    float3 out;
    out.x = fma(0.29956550340058319f, a.x,
    fma(0.63373087833825936f, a.y,
    fma(0.077705617820981968f, a.z,
    1.7557483643287353f)));

    out.y = fma(0.22158691104574774f, a.x,
    fma(0.69391388044116142f, a.y,
    fma(0.0987313588422f, a.z,
    1.7557483643287353f)));

    out.z = fma(0.02f, a.x,
    fma(0.02f, a.y,
    fma(0.20480129041026129f, a.z,
    12.226454707163354f)));

    if (clamp){
        out.x = max(out.x, 1.7557483643287353f);
        out.y = max(out.y, 1.7557483643287353f);
        out.z = max(out.z, 12.226454707163354f);
    }

    return out;
}

// The opsin dynamics XYB of a pixel from its linear RGB src and the blur of it, both before the intensity multiplier:
// the code of the former opsinDynamicsImage kernel, which wrote these 3 values over its source planes.
public float3 opsinDynamics(float3 src, float3 blurred, float intensity_multiplier){
    float3 sensitivity;
    src *= intensity_multiplier;
    blurred *= intensity_multiplier;
    blurred = butterOpsinAbsorbance(blurred, true);
    blurred = max(blurred, 1e-4f);
    sensitivity.x = divide(gamma(blurred.x), blurred.x);
    sensitivity.y = divide(gamma(blurred.y), blurred.y);
    sensitivity.z = divide(gamma(blurred.z), blurred.z);
    sensitivity = max(sensitivity, 1e-4f);
    src = butterOpsinAbsorbance(src, false);
    src *= sensitivity;
    src.x = max(src.x, 1.7557483643287353f);
    src.y = max(src.y, 1.7557483643287353f);
    src.z = max(src.z, 12.226454707163354f);

    //make positive + export
    return float3(src.x - src.y, src.x + src.y, src.z);
}
