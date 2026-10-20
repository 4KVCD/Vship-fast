module opsinModule;

// Butteraugli's opsin dynamics (libjxl's OpsinDynamicsImage), from the image and its blur (window 3, sigma 1.2)
public float gamma(float v) {
    return fma(19.245013259874995f, log(v + 9.9710635769299145), -23.16046239805755f);
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
    sensitivity.x = gamma(blurred.x) / blurred.x;
    sensitivity.y = gamma(blurred.y) / blurred.y;
    sensitivity.z = gamma(blurred.z) / blurred.z;
    sensitivity = max(sensitivity, 1e-4f);
    src = butterOpsinAbsorbance(src, false);
    src *= sensitivity;
    src.x = max(src.x, 1.7557483643287353f);
    src.y = max(src.y, 1.7557483643287353f);
    src.z = max(src.z, 12.226454707163354f);

    //make positive + export
    return float3(src.x - src.y, src.x + src.y, src.z);
}
