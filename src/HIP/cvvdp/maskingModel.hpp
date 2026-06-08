#pragma once

namespace cvvdp{

__global__ void preGaussianPreCompute_kernel(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float ch_gain = (channel == 1) ? 1.45f : 1.f;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[thid], band, channel);
    p1[thid] *= S * ch_gain;
    p2[thid] *= S * ch_gain;
    //if (thid == 0) printf("preGaussian ends with %f %f using sensi %f\n", p1[thid], p2[thid], S);
}

void preGaussianPreCompute(float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    preGaussianPreCompute_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(Lbkg, p1, p2, width, height, channel, band, csfhandle);
    GPU_CHECK(hipGetLastError());
}

//works for a band  but does all channels. It writes result on source ref
__global__ void computeD_Kernel(float* out0, float* out1, float* out2, float* out3, float* R0, float* R1, float* R2, float* R3, float* T0, float* T1, float* T2, float* T3, const int width, const int height, GaussianHandle gaussianhandle){
    const int64_t x = (threadIdx.x + blockIdx.x*gaussianThx*gaussianTileX);
    const int64_t y = (threadIdx.y + blockIdx.y*gaussianThy*gaussianTileY);

    float* gaussiankernel = gaussianhandle.gaussiankernel_d;
    float* gaussiankernel_integral = gaussianhandle.gaussiankernel_integral_d;
    __shared__ float sharedmem[tamponWidth*tamponHeight];

    const float powmaskc = powf(10, mask_c);

    GaussianSmartSharedLoadMinAbs(sharedmem, R0, T0, x, y, width, height);
    float blurred_Cm0[gaussianTileX*gaussianTileY];
    GaussianSmart_Device(blurred_Cm0, sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    #pragma unroll
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        blurred_Cm0[i] = powf(powmaskc*abs(blurred_Cm0[i]), mask_q[0]);
    }

    GaussianSmartSharedLoadMinAbs(sharedmem, R1, T1, x, y, width, height);
    float blurred_Cm1[gaussianTileX*gaussianTileY];
    GaussianSmart_Device(blurred_Cm1, sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    #pragma unroll
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        blurred_Cm1[i] = powf(powmaskc*abs(blurred_Cm1[i]), mask_q[1]);
    }

    GaussianSmartSharedLoadMinAbs(sharedmem, R2, T2, x, y, width, height);
    float blurred_Cm2[gaussianTileX*gaussianTileY];
    GaussianSmart_Device(blurred_Cm2, sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    #pragma unroll
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        blurred_Cm2[i] = powf(powmaskc*abs(blurred_Cm2[i]), mask_q[2]);
    }

    GaussianSmartSharedLoadMinAbs(sharedmem, R3, T3, x, y, width, height);
    float blurred_Cm3[gaussianTileX*gaussianTileY];
    GaussianSmart_Device(blurred_Cm3, sharedmem, x, y, width, height, gaussiankernel, gaussiankernel_integral);
    #pragma unroll
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        blurred_Cm3[i] = powf(powmaskc*abs(blurred_Cm3[i]), mask_q[3]);
    }

    #pragma unroll
    for (int i = 0; i < gaussianTileX*gaussianTileY; i++){
        const float Cmask0 = fmaf(blurred_Cm0[i], powf(2.f, xcm_weights[0]), fmaf(blurred_Cm1[i], powf(2.f, xcm_weights[4]), fmaf(blurred_Cm2[i], powf(2.f, xcm_weights[8]), blurred_Cm3[i]*powf(2.f, xcm_weights[12]))));
        const float Cmask1 = fmaf(blurred_Cm0[i], powf(2.f, xcm_weights[1]), fmaf(blurred_Cm1[i], powf(2.f, xcm_weights[5]), fmaf(blurred_Cm2[i], powf(2.f, xcm_weights[9]), blurred_Cm3[i]*powf(2.f, xcm_weights[13]))));
        const float Cmask2 = fmaf(blurred_Cm0[i], powf(2.f, xcm_weights[2]), fmaf(blurred_Cm1[i], powf(2.f, xcm_weights[6]), fmaf(blurred_Cm2[i], powf(2.f, xcm_weights[10]), blurred_Cm3[i]*powf(2.f, xcm_weights[14]))));
        const float Cmask3 = fmaf(blurred_Cm0[i], powf(2.f, xcm_weights[3]), fmaf(blurred_Cm1[i], powf(2.f, xcm_weights[7]), fmaf(blurred_Cm2[i], powf(2.f, xcm_weights[11]), blurred_Cm3[i]*powf(2.f, xcm_weights[15]))));

        //if (id == 0) printf("Cmask: %f %f %f %f from %f %f %f %f\n", Cmask0, Cmask1, Cmask2, Cmask3, blurred_Cm0, blurred_Cm1, blurred_Cm2, blurred_Cm3);
        const int region_x = i%gaussianTileX;
        const int region_y = i/gaussianTileX;
        if (x+gaussianThx*region_x >= width || y+gaussianThy*region_y >= height) return;
        const int64_t id = (y+gaussianThy*region_y)*width + x+gaussianThx*region_x;
        const float Du0 = powf(abs(R0[id] - T0[id]), mask_p)/(1+Cmask0);
        const float Du1 = powf(abs(R1[id] - T1[id]), mask_p)/(1+Cmask1);
        const float Du2 = powf(abs(R2[id] - T2[id]), mask_p)/(1+Cmask2);
        const float Du3 = powf(abs(R3[id] - T3[id]), mask_p)/(1+Cmask3);

        //if (id == 0) printf("D width %d: %f %f %f %f vs %f %f %f %f with mask %f %f %f %f\n", width, R0[id], R1[id], R2[id], R3[id], T0[id], T1[id], T2[id], T3[id], Cmask0, Cmask1, Cmask2, Cmask3);
        //if (id == 13*1024 + 64 && width == 827) printf("D width %d: %f %f %f %f\n", width, Du0, Du1, Du2, Du3);

        const float max_v = powf(10, d_max);
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
}

void computeD(float* out0, float* out1, float* out2, float* out3, float* R0, float* R1, float* R2, float* R3, float* T0, float* T1, float* T2, float* T3, const int width, const int height, GaussianHandle& gaussianhandle, hipStream_t stream){
    int th_x = gaussianThx;
    int th_y = gaussianThy;
    int bl_x = (width-1)/(th_x*gaussianTileX)+1;
    int bl_y = (height-1)/(th_y*gaussianTileY)+1;
    computeD_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(out0, out1, out2, out3, R0, R1, R2, R3, T0, T1, T2, T3, width, height, gaussianhandle);
    GPU_CHECK(hipGetLastError());
}

__global__ void computeD_baseband_kernel(float* out, float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler csfhandle){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width*height) return;

    const float S = csfhandle.computeSensitivityGPU(Lbkg[0], band, channel);
    //if (thid == 0) printf("ComputeDbasedband D: %f, T_f %f R_f %f S %f Lbkg %f\n", abs(p1[thid] - p2[thid])*S, p2[thid], p1[thid], S, Lbkg[0]);
    out[thid] = abs(p1[thid] - p2[thid])*S * baseband_weight[channel]; //the weight is put here to avoid another kernel later doing that
}

void computeD_baseband(float* out, float* Lbkg, float* p1, float* p2, int width, int height, int channel, int band, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;
    computeD_baseband_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(out, Lbkg, p1, p2, width, height, channel, band, csfhandle);
    GPU_CHECK(hipGetLastError());
}

}
