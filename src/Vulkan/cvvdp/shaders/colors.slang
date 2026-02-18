float3 xyz_to_LMS2006(float3 a){
    float3 out;

    out.x = fma(0.187596268556126, a.x, fma(0.585168649077728, a.y, -0.026384263306304*a.z));
    out.y = fma(-0.133397430663221, a.x, fma(0.405505777260049, a.y, 0.034502127690364*a.z));
    out.z = fma(0.000244379021663, a.x, fma(-0.000542995890619, a.y, 0.019406849066323*a.z));

    return out;
}

float3 LMS2006_to_DKLd65(float3 a){
    //a = x+y
    a.x = a.x + a.y;
    //b = x -2.311130179947035y = a -3.311130179947035y
    a.y = a.x - 3.311130179947035*a.y;
    //c = -x-y + 50.977571328718781z = -a + 50.977571328718781z
    a.z = 50.977571328718781*a.z - a.x;
    return a;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void XYZ_to_dkl(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p1, uniform float* p2, uniform float* p3, uniform int64_t width){
    const int64_t x = threadIdx.x + blockIdx.x * 256;
    if (x >= width) return;

    float3 src = {p1[x], p2[x], p3[x]};

    //linrgb_to_xyz(src);
    src = xyz_to_LMS2006(src);
    src = LMS2006_to_DKLd65(src);

    //if (x == 0) printf("value at DKL / Linrgb %f %f %f %f %f %f\n", src.x, src.y, src.z, p1[x], p2[x], p3[x]);

    p1[x] = src.x; p2[x] = src.y; p3[x] = src.z;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void displayEncodeBT709(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p, uniform int64_t width, uniform float Y_peak, uniform float Y_black, uniform float Y_refl, uniform float exposure){
    const int64_t x = threadIdx.x + blockIdx.x * 256;
    if (x >= width) return;

    p[x] = (Y_peak - Y_black) * (max(0.f, min(1.f, p[x]*exposure))) + Y_black + Y_refl;
    //if (x == 0) printf("We got %f\n", p[x]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void displayEncodePQ(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p, uniform int64_t width, uniform float Y_peak, uniform float Y_black, uniform float Y_refl, uniform float exposure){
    const int64_t x = threadIdx.x + blockIdx.x * 256;
    if (x >= width) return;

    //convert the Y normalized to 100 to actual PQ luminosity values
    p[x] = max(max(0.005f, Y_black), min(Y_peak, p[x]*100.f*exposure)) + Y_black + Y_refl;
    //if (x == 0) printf("We got %f\n", p[x]);
}