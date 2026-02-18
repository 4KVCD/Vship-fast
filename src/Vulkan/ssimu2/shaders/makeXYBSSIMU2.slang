float3 opsin_absorbance(float3 a){
    float3 out;
    const float opsin_bias = 0.0037930734f;
    out.x = fma(0.30f, a.x,
    fma(0.622f, a.y,
    fma(0.078f, a.z,
    opsin_bias)));

    out.y = fma(0.23f, a.x,
    fma(0.692f, a.y,
    fma(0.078f, a.z,
    opsin_bias)));

    out.z = fma(0.24342269f, a.x,
    fma(0.20476745f, a.y,
    fma(0.55180986f, a.z,
    opsin_bias)));

    return out;
}

float3 mixed_to_xyb(float3 a){
    a.x = 0.5f * (a.x - a.y);
    a.y = a.x + a.y;
    return a;
}

float3 linear_rgb_to_xyb(float3 a){
    const float abs_bias = -0.1559542025327239f;
    a = opsin_absorbance(a);
    //printf("from %f to %f\n", a.x, cbrtf(a.x*((int)(a.x >= 0))));
    a.x = pow(a.x * ((int)(a.x >= 0)), 0.333f) + abs_bias;
    a.y = pow(a.y * ((int)(a.y >= 0)), 0.333f) + abs_bias;
    a.z = pow(a.z * ((int)(a.z >= 0)), 0.333f) + abs_bias;
    //printf("got %f, %f, %f\n", a.x, a.y, a.z);
    a = mixed_to_xyb(a);
    return a;
}

float3 make_positive_xyb(float3 a) {
    a.z = (a.z - a.y) + 0.55;
    a.x = a.x * 14.0f + 0.42;  
    a.y += 0.01;
    return a;
}

float3 rgb_to_positive_xyb_d(float3 a){
    a = linear_rgb_to_xyb(a);
    a = make_positive_xyb(a);
    return a;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void rgb_to_positive_xyb(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p0, uniform float* p1, uniform float* p2, uniform int64_t width){
    int64_t x = threadIdx.x + blockIdx.x*256;
    if (x >= width) return;
    float3 packed = {p0[x], p1[x], p2[x]};
    packed = rgb_to_positive_xyb_d(packed);
    p0[x] = packed.x;
    p1[x] = packed.y;
    p2[x] = packed.z;
    //if (x == 10000) printf("XYB at 10000 : %f %f %f\n", packed.x, packed.y, packed.z);
}