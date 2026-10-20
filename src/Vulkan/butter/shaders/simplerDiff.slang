import xybLowFreqModule;

[shader("compute")]
[numthreads(256, 1, 1)]
void samenoisediff(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* dst, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    dst[x] += w * src1[x] * src1[x];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void diffclamp(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* dst, uniform uint64_t width, uniform float maxclamp){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    float v0 = abs(src1[x]);
    float v1 = abs(src2[x]);

    if (v0 > maxclamp) v0 = maxclamp;
    if (v1 > maxclamp) v1 = maxclamp;
    dst[x] = v1 - v0;
}

// libjxl's L2 differences (lib/jxl/butteraugli/butteraugli.cc), with its fused multiply-adds (MulAdd); every other
// operation is rounded on its own.

// L2Diff: dst + (src1 - src2)^2 * w
float l2diff(float src1, float src2, float dst, float w){
    const float diff = src1 - src2;
    return fma(diff*diff, w, dst);
}

// SetL2Diff: (src1 - src2)^2 * w
float setL2diff(float src1, float src2, float w){
    const float diff = src1 - src2;
    return (diff*diff)*w;
}

// L2DiffAsymmetric: dst + the asymmetric objectives of val0 (reference) and val1, w_0gt1 and w_0lt1 already scaled by
// 0.8 (in double, as libjxl does)
float l2AsymDiff(float val0, float val1, float dst, float w_0gt1, float w_0lt1){
    // primary symmetric quadratic objective
    const float diff = val0 - val1;
    const float total = fma(diff*diff, w_0gt1, dst);

    // secondary half-open quadratic objectives
    const float fabs0 = abs(val0);
    const float too_small = 0.4f*fabs0;
    const float too_big = fabs0;
    float v = 0.0f;
    if (val0 < 0.0f){
        if (val1 > -too_small) v = val1 + too_small;
        else if (val1 < -too_big) v = -val1 - too_big;
    } else {
        if (val1 < too_small) v = too_small - val1;
        else if (val1 > too_big) v = val1 - too_big;
    }
    return fma(w_0lt1, v*v, total);
}
// libjxl's Mask adds the difference of the two blurred masks to block_diff_ac (Y channel):
// diff_ac += kMaskToErrorMul * diff * diff, with w = kMaskToErrorMul = 10, one C++ expression that its compiler fuses
[shader("compute")]
[numthreads(256, 1, 1)]
void L2diff(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* dst, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float diff = src1[x] - src2[x];
    dst[x] = fma(w*diff, diff, dst[x]);
}
// placeGetdiffmap's L2 differences in one pass, in libjxl's order, for the 3 channels c:
// - block_diff_ac[c] gets L2DiffAsymmetric of the high frequencies (c < 2), then L2Diff of the medium ones;
// - block_diff_dc[c] gets SetL2Diff of the low ones, converted by XybLowFreqToVals first (which separateFrequencies
//   used to do in place in a pass of its own).
// planes holds placeGetdiffmap's planes (plane k at planes + k*width): lf1 3-5, mf1 6-8, hf1 9-10, lf2 13-15,
// mf2 16-18, hf2 19-20; ac and dc hold 3 planes each.
[shader("compute")]
[numthreads(256, 1, 1)]
void L2diffs(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* planes, uniform float* ac, uniform float* dc, uniform uint64_t width, uniform float asym0gt1X, uniform float asym0lt1X, uniform float asym0gt1Y, uniform float asym0lt1Y, uniform float mfX, uniform float mfY, uniform float mfB, uniform float lfX, uniform float lfY, uniform float lfB){
    const uint64_t x = threadIdx.x + blockIdx.x*256;
    if (x >= width) return;
    const float asymWeights[4] = {asym0gt1X, asym0lt1X, asym0gt1Y, asym0lt1Y};
    const float mfWeights[3] = {mfX, mfY, mfB};
    const float lfWeights[3] = {lfX, lfY, lfB};

    // every load before the first store: as far as the compiler knows the stores could alias the planes, so it
    // would not move a later load above them
    float acIn[3], hf1[2], hf2[2], mf1[3], mf2[3];
    float3 lf1, lf2;
    [ForceUnroll]
    for (int c = 0; c < 3; c++){
        acIn[c] = ac[c*width + x];
        if (c < 2){
            hf1[c] = planes[(9+c)*width + x];
            hf2[c] = planes[(19+c)*width + x];
        }
        mf1[c] = planes[(6+c)*width + x];
        mf2[c] = planes[(16+c)*width + x];
        lf1[c] = planes[(3+c)*width + x];
        lf2[c] = planes[(13+c)*width + x];
    }
    lf1 = XybLowFreqToVals_Device(lf1);
    lf2 = XybLowFreqToVals_Device(lf2);
    [ForceUnroll]
    for (int c = 0; c < 3; c++){
        float acc = acIn[c];
        if (c < 2) acc = l2AsymDiff(hf1[c], hf2[c], acc, asymWeights[2*c], asymWeights[2*c+1]);
        ac[c*width + x] = l2diff(mf1[c], mf2[c], acc, mfWeights[c]);
        dc[c*width + x] = setL2diff(lf1[c], lf2[c], lfWeights[c]);
    }
}