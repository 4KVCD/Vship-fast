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

[shader("compute")]
[numthreads(256, 1, 1)]
void L2diff(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* src1, uniform float* src2, uniform float* dst, uniform uint64_t width, uniform float w){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    const float diff = src1[x] - src2[x];
    dst[x] += w * diff * diff;
    //if (x == 10000) printf("l2diff_dc : %f from %f, %f\n", dst[x], src1[x], src2[x]);
}

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