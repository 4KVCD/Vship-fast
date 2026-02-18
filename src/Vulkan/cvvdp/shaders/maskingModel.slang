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

static groupshared float tampon[32*32];

//tampon should be groupshared memory of size 32*32
void GaussianSmartSharedLoadMinAbs(const int thx, const int thy, float* src1, float* src2, int64_t x, int64_t y, int64_t width, int64_t height){
    const int64_t tampon_base_x = x - thx - 8;
    const int64_t tampon_base_y = y - thy - 8;

    //fill tampon
    //fill tampon
    tampon[thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? min(abs(src1[(tampon_base_y+thy)*width + tampon_base_x+thx]), abs(src2[(tampon_base_y+thy)*width + tampon_base_x+thx])) : 0.f;
    tampon[(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? min(abs(src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx]), abs(src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx])) : 0.f;
    tampon[thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? min(abs(src1[(tampon_base_y+thy)*width + tampon_base_x+thx+16]), abs(src2[(tampon_base_y+thy)*width + tampon_base_x+thx+16])) : 0.f;
    tampon[(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? min(abs(src1[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16]), abs(src2[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16])) : 0.f;
    GroupMemoryBarrierWithGroupSync();
}

//a whole block of 16x16 should into there, x and y corresponds to their real position in the src (or slighly outside)
//at the end, the central 16*16 part of tampon contains the blurred value for each thread
//tampon is of size 32*32
float GaussianSmart_Device(const int thx, const int thy, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    //horizontalBlur on tampon restraint into rectangle [8 - 24][0 - 32] -> 2 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot;
    float out = 0.f;
    float out2 = 0.f;
    //border handling precompute
    int64_t beg = max((int64_t)0, x-8)-(x-8);
    int64_t end2 = min(width, x+9)-min(width, (x-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[thy*32 + thx+i]*gaussiankernel[i];
        //2nd pass in [8 - 24][16 - 32]
        out2 += tampon[(thy+16)*32 + thx+i]*gaussiankernel[i];
    }
    GroupMemoryBarrierWithGroupSync();
    tampon[thy*32 + thx+8] = out/tot;
    tampon[(thy+16)*32 + thx+8] = out2/tot;
    GroupMemoryBarrierWithGroupSync();

    //verticalBlur on tampon[tamponOffset] restraint into rectangle [8 - 24][8 - 24] -> 1 pass per thread
    out = 0.f;
    beg = max((int64_t)0, y-8)-(y-8);
    end2 = min(height, y+9)-min(height, (y-8));
    tot = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[(thy+i)*32 + thx+8]*gaussiankernel[i];
    }

    GroupMemoryBarrierWithGroupSync();
    return out/tot;
}

//works for a band  but does all channels. It writes result on source ref
//work on 16x16 blocks
[shader("compute")]
[numthreads(16, 16, 1)]
void computeD(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* R0, uniform float* R1, uniform float* R2, uniform float* R3, uniform float* T0, uniform float* T1, uniform float* T2, uniform float* T3, uniform int64_t width, uniform int64_t height, uniform float* gaussiankernel, uniform float* gaussiankernel_integral){
    const int64_t x = (threadIdx.x + blockIdx.x*16);
    const int64_t y = (threadIdx.y + blockIdx.y*16);
    const int64_t id = y*width + x;

    const float powmaskc = pow(10, mask_c);

    GaussianSmartSharedLoadMinAbs(threadIdx.x, threadIdx.y, R0, T0, x, y, width, height);
    const float blurred0 = GaussianSmart_Device(threadIdx.x, threadIdx.y, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm0 = pow(powmaskc*abs(blurred0), mask_q[0]);

    GaussianSmartSharedLoadMinAbs(threadIdx.x, threadIdx.y, R1, T1, x, y, width, height);
    const float blurred1 = GaussianSmart_Device(threadIdx.x, threadIdx.y, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm1 = pow(powmaskc*abs(blurred1), mask_q[1]);

    GaussianSmartSharedLoadMinAbs(threadIdx.x, threadIdx.y, R2, T2, x, y, width, height);
    const float blurred2 = GaussianSmart_Device(threadIdx.x, threadIdx.y, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm2 = pow(powmaskc*abs(blurred2), mask_q[2]);

    GaussianSmartSharedLoadMinAbs(threadIdx.x, threadIdx.y, R3, T3, x, y, width, height);
    const float blurred3 = GaussianSmart_Device(threadIdx.x, threadIdx.y, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    const float blurred_Cm3 = pow(powmaskc*abs(blurred3), mask_q[3]);

    const float Cmask0 = fma(blurred_Cm0, pow(2.f, xcm_weights[0]), fma(blurred_Cm1, pow(2.f, xcm_weights[4]), fma(blurred_Cm2, pow(2.f, xcm_weights[8]), blurred_Cm3*pow(2.f, xcm_weights[12]))));
    const float Cmask1 = fma(blurred_Cm0, pow(2.f, xcm_weights[1]), fma(blurred_Cm1, pow(2.f, xcm_weights[5]), fma(blurred_Cm2, pow(2.f, xcm_weights[9]), blurred_Cm3*pow(2.f, xcm_weights[13]))));
    const float Cmask2 = fma(blurred_Cm0, pow(2.f, xcm_weights[2]), fma(blurred_Cm1, pow(2.f, xcm_weights[6]), fma(blurred_Cm2, pow(2.f, xcm_weights[10]), blurred_Cm3*pow(2.f, xcm_weights[14]))));
    const float Cmask3 = fma(blurred_Cm0, pow(2.f, xcm_weights[3]), fma(blurred_Cm1, pow(2.f, xcm_weights[7]), fma(blurred_Cm2, pow(2.f, xcm_weights[11]), blurred_Cm3*pow(2.f, xcm_weights[15]))));
    //if (id == 0) printf("Cmask: %f %f %f %f from %f %f %f %f\n", Cmask0, Cmask1, Cmask2, Cmask3, blurred_Cm0, blurred_Cm1, blurred_Cm2, blurred_Cm3);

    if (x >= width || y >= height) return;
    const float Du0 = pow(abs(R0[id] - T0[id]), mask_p)/(1+Cmask0);
    const float Du1 = pow(abs(R1[id] - T1[id]), mask_p)/(1+Cmask1);
    const float Du2 = pow(abs(R2[id] - T2[id]), mask_p)/(1+Cmask2);
    const float Du3 = pow(abs(R3[id] - T3[id]), mask_p)/(1+Cmask3);

    //if (id == 0) printf("D width %d: %f %f %f %f vs %f %f %f %f with mask %f %f %f %f\n", width, R0[id], R1[id], R2[id], R3[id], T0[id], T1[id], T2[id], T3[id], Cmask0, Cmask1, Cmask2, Cmask3);
    //if (id == 13*1024 + 64 && width == 827) printf("D width %d: %f %f %f %f\n", width, Du0, Du1, Du2, Du3);

    const float max_v = pow(10, d_max);
    const float D0 = max_v*Du0/(max_v+Du0);
    const float D1 = max_v*Du1/(max_v+Du1);
    const float D2 = max_v*Du2/(max_v+Du2);
    const float D3 = max_v*Du3/(max_v+Du3);

    //we write D inside Reference planes along with the channel multiplier
    R0[id] = D0 * 1.f;
    R1[id] = D1 * ch_chrom_w;
    R2[id] = D2 * ch_chrom_w;
    R3[id] = D3 * ch_trans_w;

    //if (id == 13*1024 + 64 && width == 827) printf("D id/width %lld/%d: %f %f %f %f\n", id, width, D0, D1, D2, D3);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeD_baseband(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* Lbkg, uniform float* p1, uniform float* p2, uniform float* CSFMem_d, uniform int width, uniform int height, uniform int channel, uniform int band){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= width*height) return;

    const float S = computeSensitivityGPU(Lbkg[0], band, channel, CSFMem_d);
    //if (thid == 0) printf("ComputeDbasedband D: %f, T_f %f R_f %f S %f Lbkg %f\n", abs(p1[thid] - p2[thid])*S, p2[thid], p1[thid], S, Lbkg[0]);
    p1[thid] = abs(p1[thid] - p2[thid])*S * baseband_weight[channel]; //the weight is put here to avoid another kernel later doing that
}