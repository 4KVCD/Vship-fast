float MaskY(float delta) {
    const float offset = 0.829591754942f;
    const float scaler = 0.451936922203f;
    const float mul = 2.5485944793f;
    const float c = mul / ((scaler * delta) + offset);
    const float retval = (1.0 + c)*(1.0/(0.79079917404f*17.83f));
    return retval * retval;
}

float MaskDcY(float delta) {
    const float offset = 0.20025578522f;
    const float scaler = 3.87449418804f;
    const float mul = 0.505054525019f;
    const float c = mul / ((scaler * delta) + offset);
    const float retval = (1.0 + c)*(1.0/(0.79079917404f*17.83f));
    return retval * retval;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeDiffmap(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* mask, uniform float* block_diff_dc0, uniform float* block_diff_dc1, uniform float* block_diff_dc2, uniform float* block_diff_ac0, uniform float* block_diff_ac1, uniform float* block_diff_ac2, uniform float* dst, uniform uint64_t width){
    uint64_t x = threadIdx.x + blockIdx.x*256;

    if (x >= width) return;

    float val = mask[x];
    float maskval = MaskY(val);
    float maskval_dc = MaskDcY(val);

    dst[x] = sqrt(maskval * (block_diff_ac0[x] + block_diff_ac1[x] + block_diff_ac2[x]) + maskval_dc * (block_diff_dc0[x] + block_diff_dc1[x] + block_diff_dc2[x]));
    //if ((x == 376098 && width == 1080*1920) || ((x == 59456 && width == 540*960))) printf("final result: %f with ac: %f, %f, %f, dc: %f, %f, %f, mask: %f.\n", dst[x], block_diff_ac0[x], block_diff_ac1[x], block_diff_ac2[x], block_diff_dc0[x], block_diff_dc1[x], block_diff_dc2[x], mask[x]);
}