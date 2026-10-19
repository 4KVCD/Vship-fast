import CSFModule;

static const float xcm_weights[16] = {
    -0.18950104713439941,
    -5.962151050567627,
    -4.31834602355957,
    -1.9321587085723877,
    2.5655593872070312,
    0.34406712651252747,
    -2.719646453857422,
    -0.4970424771308899,
    3.8118371963500977,
    -1.0051705837249756,
    -0.5193376541137695,
    -0.5653647780418396,
    -7.054771423339844,
    -5.527150630950928,
    -3.5106418132781982,
    -2.08804988861084,
};
static const float baseband_weight[4] = {
    0.0036334486212581396,
    1.6627724170684814,
    4.11874532699585,
    25.25969886779785
};
static const float ch_chrom_w = 1.;
static const float ch_trans_w = 0.8081134557723999;
static const float d_max = 2.5642454624176025;
static const float mask_p = 2.264355182647705;
static const float mask_c = -0.7954971194267273;
static const float mask_q[4] = {
    1.302622675895691,
    2.8885908126831055,
    3.6807713508605957,
    3.588787317276001,
};

[shader("compute")]
[numthreads(256, 1, 1)]
void preGaussianPreCompute(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* Lbkg, uniform float* p1, uniform float* p2, uniform float* CSFMem_d, uniform int64_t width, uniform int64_t height, uniform int channel, uniform int band){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= width*height) return;

    const float ch_gain = (channel == 1) ? 1.45f : 1.f;

    const float S = computeSensitivityGPU(Lbkg[thid], band, channel, CSFMem_d);
    p1[thid] *= S * ch_gain;
    p2[thid] *= S * ch_gain;
    //if (thid == 0) printf("preGaussian ends with %f %f using sensi %f\n", p1[thid], p2[thid], S);
}

static groupshared float tampon[4][32*32]; //one tile per channel
static const int PU_RADIUS = 6; //puRadius in gaussianBlur.hpp
static const int PU_PADSIZE = 6; //pycvvdp blurs only bands larger than this in both directions

// pycvvdp's safe_pow: (x + 1e-5)^p - 1e-5^p. The GPU's pow may round x = 0 slightly below the
// constant, so the result is clamped at 0 (a negative value would make the pooling's pow NaN).
float safe_pow(float x, float p){
    const float epsilon = 0.00001f;
    return max(pow(x + epsilon, p) - pow(epsilon, p), 0.f);
}

// torch's reflect padding: -1 -> 1, size -> size-2
int64_t reflect(int64_t x, int64_t size){
    if (size == 1) return 0; //a single pixel reflects onto itself
    while (x < 0 || x >= size){
        x = (x < 0) ? -x : 2*size - 2 - x;
    }
    return x;
}

// Loads min(|src1|, |src2|) of channel c for this 16x16 block and a halo of 8 pixels into tampon[c], reflected at
// the borders.
void GaussianSmartSharedLoadMinAbs(const int c, const int thx, const int thy, float* src1, float* src2, int64_t x, int64_t y, int64_t width, int64_t height){
    const int64_t tampon_base_x = x - thx - 8;
    const int64_t tampon_base_y = y - thy - 8;
    for (int dy = 0; dy < 32; dy += 16){
        for (int dx = 0; dx < 32; dx += 16){
            const int64_t pixel = reflect(tampon_base_y + thy + dy, height)*width + reflect(tampon_base_x + thx + dx, width);
            tampon[c][(thy+dy)*32 + thx+dx] = min(abs(src1[pixel]), abs(src2[pixel]));
        }
    }
}

// pycvvdp's phase uncertainty blur of the values loaded for this thread's pixel in the 4 channels: the 13-tap
// kernel horizontally then vertically. Bands of 6 pixels or less in either direction are not blurred, like pycvvdp.
float4 GaussianSmart_Device(const int thx, const int thy, int64_t width, int64_t height, float* gaussiankernel){
    float4 result;
    if (width <= PU_PADSIZE || height <= PU_PADSIZE){
        [ForceUnroll]
        for (int c = 0; c < 4; c++) result[c] = tampon[c][(thy+8)*32 + thx+8];
        return result;
    }
    float out[4], out2[4];
    [ForceUnroll]
    for (int c = 0; c < 4; c++){
        out[c] = 0.f;
        out2[c] = 0.f;
        for (int i = -PU_RADIUS; i <= PU_RADIUS; i++){
            out[c] += tampon[c][thy*32 + thx+8+i]*gaussiankernel[i+PU_RADIUS];
            out2[c] += tampon[c][(thy+16)*32 + thx+8+i]*gaussiankernel[i+PU_RADIUS];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (int c = 0; c < 4; c++){
        tampon[c][thy*32 + thx+8] = out[c];
        tampon[c][(thy+16)*32 + thx+8] = out2[c];
    }
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll]
    for (int c = 0; c < 4; c++){
        float sum = 0.f;
        for (int i = -PU_RADIUS; i <= PU_RADIUS; i++){
            sum += tampon[c][(thy+8+i)*32 + thx+8]*gaussiankernel[i+PU_RADIUS];
        }
        result[c] = sum;
    }
    return result;
}

//works for a band  but does all channels. It writes result on source ref
//work on 16x16 blocks
[shader("compute")]
[numthreads(16, 16, 1)]
void computeD(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* out0, uniform float* out1, uniform float* out2, uniform float* out3, uniform float* R0, uniform float* R1, uniform float* R2, uniform float* R3, uniform float* T0, uniform float* T1, uniform float* T2, uniform float* T3, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel){
    const int64_t x = (threadIdx.x + blockIdx.x*16);
    const int64_t y = (threadIdx.y + blockIdx.y*16);
    const int64_t id = y*width + x;

    const float powmaskc = pow(10, mask_c);

    //the 4 channels are loaded and blurred together, so that their reads overlap and they share the barriers
    GaussianSmartSharedLoadMinAbs(0, threadIdx.x, threadIdx.y, R0, T0, x, y, width, height);
    GaussianSmartSharedLoadMinAbs(1, threadIdx.x, threadIdx.y, R1, T1, x, y, width, height);
    GaussianSmartSharedLoadMinAbs(2, threadIdx.x, threadIdx.y, R2, T2, x, y, width, height);
    GaussianSmartSharedLoadMinAbs(3, threadIdx.x, threadIdx.y, R3, T3, x, y, width, height);
    GroupMemoryBarrierWithGroupSync();
    const float4 blurred = GaussianSmart_Device(threadIdx.x, threadIdx.y, width, height, gaussiankernel);
    const float blurred_Cm0 = safe_pow(powmaskc*abs(blurred[0]), mask_q[0]);
    const float blurred_Cm1 = safe_pow(powmaskc*abs(blurred[1]), mask_q[1]);
    const float blurred_Cm2 = safe_pow(powmaskc*abs(blurred[2]), mask_q[2]);
    const float blurred_Cm3 = safe_pow(powmaskc*abs(blurred[3]), mask_q[3]);

    const float Cmask0 = fma(blurred_Cm0, pow(2.f, xcm_weights[0]), fma(blurred_Cm1, pow(2.f, xcm_weights[4]), fma(blurred_Cm2, pow(2.f, xcm_weights[8]), blurred_Cm3*pow(2.f, xcm_weights[12]))));
    const float Cmask1 = fma(blurred_Cm0, pow(2.f, xcm_weights[1]), fma(blurred_Cm1, pow(2.f, xcm_weights[5]), fma(blurred_Cm2, pow(2.f, xcm_weights[9]), blurred_Cm3*pow(2.f, xcm_weights[13]))));
    const float Cmask2 = fma(blurred_Cm0, pow(2.f, xcm_weights[2]), fma(blurred_Cm1, pow(2.f, xcm_weights[6]), fma(blurred_Cm2, pow(2.f, xcm_weights[10]), blurred_Cm3*pow(2.f, xcm_weights[14]))));
    const float Cmask3 = fma(blurred_Cm0, pow(2.f, xcm_weights[3]), fma(blurred_Cm1, pow(2.f, xcm_weights[7]), fma(blurred_Cm2, pow(2.f, xcm_weights[11]), blurred_Cm3*pow(2.f, xcm_weights[15]))));
    //if (id == 0) printf("Cmask: %f %f %f %f from %f %f %f %f\n", Cmask0, Cmask1, Cmask2, Cmask3, blurred_Cm0, blurred_Cm1, blurred_Cm2, blurred_Cm3);

    if (x >= width || y >= height) return;
    const float Du0 = safe_pow(abs(R0[id] - T0[id]), mask_p)/(1+Cmask0);
    const float Du1 = safe_pow(abs(R1[id] - T1[id]), mask_p)/(1+Cmask1);
    const float Du2 = safe_pow(abs(R2[id] - T2[id]), mask_p)/(1+Cmask2);
    const float Du3 = safe_pow(abs(R3[id] - T3[id]), mask_p)/(1+Cmask3);

    //if (id == 0) printf("D width %d: %f %f %f %f vs %f %f %f %f with mask %f %f %f %f\n", width, R0[id], R1[id], R2[id], R3[id], T0[id], T1[id], T2[id], T3[id], Cmask0, Cmask1, Cmask2, Cmask3);
    //if (id == 13*1024 + 64 && width == 827) printf("D width %d: %f %f %f %f\n", width, Du0, Du1, Du2, Du3);

    const float max_v = pow(10, d_max);
    const float D0 = max_v*Du0/(max_v+Du0);
    const float D1 = max_v*Du1/(max_v+Du1);
    const float D2 = max_v*Du2/(max_v+Du2);
    const float D3 = max_v*Du3/(max_v+Du3);

    //we write D inside Reference planes along with the channel multiplier
    out0[id] = D0 * 1.f;
    out1[id] = D1 * ch_chrom_w;
    out2[id] = D2 * ch_chrom_w;
    out3[id] = D3 * ch_trans_w;

    //if (id == 13*1024 + 64 && width == 827) printf("D id/width %lld/%d: %f %f %f %f\n", id, width, D0, D1, D2, D3);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeD_baseband(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* out, uniform float* Lbkg, uniform float* p1, uniform float* p2, uniform float* CSFMem_d, uniform int width, uniform int height, uniform int channel, uniform int band){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= width*height) return;

    const float S = computeSensitivityGPU(Lbkg[0], band, channel, CSFMem_d);
    //if (thid == 0) printf("ComputeDbasedband D: %f, T_f %f R_f %f S %f Lbkg %f\n", abs(p1[thid] - p2[thid])*S, p2[thid], p1[thid], S, Lbkg[0]);
    out[thid] = abs(p1[thid] - p2[thid])*S * baseband_weight[channel]; //the weight is put here to avoid another kernel later doing that
}