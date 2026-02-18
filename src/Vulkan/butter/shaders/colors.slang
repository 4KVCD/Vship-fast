float gamma(float v) {
    return fma(19.245013259874995f, log(v + 9.9710635769299145), -23.16046239805755f);
}

float3 butterOpsinAbsorbance(float3 a, bool clamp = false){
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

[shader("compute")]
[numthreads(256, 1, 1)]
void opsinDynamicsImage(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* src3, uniform float* blurred1, uniform float* blurred2, uniform float* blurred3, uniform uint64_t width, uniform uint64_t height, uniform float intensity_multiplier){
    uint64_t x = threadIdx.x + blockIdx.x*256;
    if (x >= width*height) return;

    float3 sensitivity;
    float3 src = {src1[x], src2[x], src3[x]};
    float3 blurred = {blurred1[x], blurred2[x], blurred3[x]};
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

    src1[x] = src.x - src.y; 
    src2[x] = src.x + src.y; 
    src3[x] = src.z;
    //if ((x == 376098 && width*height == 1080*1920)) printf("%f, %f, %f and %f, %f, %f to %f, %f, %f with %f, %f, %f sens\n", oldsrc.x, oldsrc.y, oldsrc.z, oldblurred.x, oldblurred.y, oldblurred.z, src1[x], src2[x], src3[x], sensitivity.x, sensitivity.y, sensitivity.z);
}