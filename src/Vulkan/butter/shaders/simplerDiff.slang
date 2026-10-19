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

// dst + w*diff^2 for diff = src1 - src2, as L2diff adds it.
float l2diff(float src1, float src2, float dst, float w){
    const float diff = src1 - src2;
    dst += w * diff * diff;
    return dst;
}

// Kept as its own kernel, word for word as it was: the drivers round its additions differently once its body is
// fused into L2diffs (NVIDIA's compiler and the AMD driver each combine some of them, in different ways), and no
// single fused form gives each GPU's former result.
[shader("compute")]
[numthreads(256, 1, 1)]
void L2AsymDiff(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* dst, uniform uint64_t width, uniform float w_0gt1, uniform float w_0lt1){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float diff = src1[x] - src2[x];
    dst[x] += w_0gt1 * diff * diff;

    const float fabs0 = abs(src1[x]);
    const float too_small = 0.4 * fabs0;
    const float too_big = fabs0;

    if (src1[x] < 0) {
        if (src2[x] > -too_small) {
            float v = src2[x] + too_small;
            dst[x] += w_0lt1 * v * v;
        } else if (src2[x] < -too_big) {
            float v = -src2[x] - too_big;
            dst[x] += w_0lt1 * v * v;
        }
    } else {
        if (src2[x] < too_small) {
            float v = too_small - src2[x];
            dst[x] += w_0lt1 * v * v;
        } else if (src2[x] > too_big) {
            float v = src2[x] - too_big;
            dst[x] += w_0lt1 * v * v;
        }
    }
}

// x unchanged, through an operation the compiler cannot see through (zero is 0), so that the float operations
// before and after it are not combined across it, as when the value went through memory between two kernels.
float opaque(float x, uint zero){
    return asfloat(asuint(x) | zero);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void L2diff(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* dst, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    dst[x] = l2diff(src1[x], src2[x], dst[x], w);
}

// placeGetdiffmap's L2 differences after L2AsymDiff, in one pass, for the 3 channels c:
// - block_diff_ac[c] gets L2diff of the medium frequencies;
// - block_diff_dc[c], which starts at 0, gets L2diff of the low ones, converted by XybLowFreqToVals first (which
//   separateFrequencies used to do in place in a pass of its own).
// Each step is the code of its former kernel, and its result goes through opaque() as it went through memory, so
// every value is the same float as with the separate kernels. planes holds placeGetdiffmap's planes (plane k at
// planes + k*width): lf1 3-5, mf1 6-8, lf2 13-15, mf2 16-18; ac and dc hold 3 planes each.
[shader("compute")]
[numthreads(256, 1, 1)]
void L2diffs(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* planes, uniform float* ac, uniform float* dc, uniform uint64_t width, uniform float mfX, uniform float mfY, uniform float mfB, uniform float lfX, uniform float lfY, uniform float lfB, uniform uint zero){
    const uint64_t x = threadIdx.x + blockIdx.x*256;
    if (x >= width) return;
    const float mfWeights[3] = {mfX, mfY, mfB};
    const float lfWeights[3] = {lfX, lfY, lfB};

    // every load before the first store: as far as the compiler knows the stores could alias the planes, so it
    // would not move a later load above them
    float acIn[3], mf1[3], mf2[3];
    float3 lf1, lf2;
    [ForceUnroll]
    for (int c = 0; c < 3; c++){
        acIn[c] = ac[c*width + x];
        mf1[c] = planes[(6+c)*width + x];
        mf2[c] = planes[(16+c)*width + x];
        lf1[c] = planes[(3+c)*width + x];
        lf2[c] = planes[(13+c)*width + x];
    }
    lf1 = XybLowFreqToVals_Device(lf1);
    lf2 = XybLowFreqToVals_Device(lf2);
    [ForceUnroll]
    for (int c = 0; c < 3; c++){
        ac[c*width + x] = l2diff(mf1[c], mf2[c], acIn[c], mfWeights[c]);
        dc[c*width + x] = l2diff(opaque(lf1[c], zero), opaque(lf2[c], zero), asfloat(zero), lfWeights[c]);
    }
}