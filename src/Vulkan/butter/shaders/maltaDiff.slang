
//this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
groupshared float diffs[24*24]; //4 around each border. 24*24 = 1.5² * 16*16 -> around 2.25 values per thread. Note that we can pad in advance with negligible register cost
//shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)

// One diffs value of the tile, dy rows and dx columns away from index o
float tap(int o, int dy, int dx) { return diffs[o + dy*24 + dx]; }

// The sum of the squared sums along 16 short lines through the pixel at index o: libjxl's MaltaUnit
// (lib/jxl/butteraugli/butteraugli.cc) with its lines, the order it adds their taps in and its fma.
// MaltaUnitLF's lines have 5 taps, MaltaUnit's 7 or 9.
float MaltaUnitLF(int o) {
    float retval;
    // x grows, y constant
    {
        const float sum = (tap(o, 0, -4) + tap(o, 0, -2)) + (tap(o, 0, 0) + (tap(o, 0, 2) + tap(o, 0, 4)));
        retval = sum*sum;
    }
    // y grows, x constant
    {
        const float sum = (tap(o, -4, 0) + tap(o, -2, 0)) + (tap(o, 0, 0) + (tap(o, 2, 0) + tap(o, 4, 0)));
        retval = fma(sum, sum, retval);
    }
    // both grow
    {
        const float sum = (tap(o, -3, -3) + tap(o, -2, -2)) + (tap(o, 0, 0) + (tap(o, 2, 2) + tap(o, 3, 3)));
        retval = fma(sum, sum, retval);
    }
    // y grows, x shrinks
    {
        const float sum = (tap(o, -3, 3) + tap(o, -2, 2)) + (tap(o, 0, 0) + (tap(o, 2, -2) + tap(o, 3, -3)));
        retval = fma(sum, sum, retval);
    }
    // y grows -4 to 4, x shrinks 1 -> -1
    {
        const float sum = (tap(o, -4, 1) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 4, -1)));
        retval = fma(sum, sum, retval);
    }
    //  y grows -4 to 4, x grows -1 -> 1
    {
        const float sum = (tap(o, -4, -1) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 4, 1)));
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y grows -1 to 1
    {
        const float sum = (tap(o, -1, -4) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 1, 4)));
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y shrinks 1 to -1
    {
        const float sum = (tap(o, 1, -4) + tap(o, 1, -2)) + (tap(o, 0, 0) + (tap(o, -1, 2) + tap(o, -1, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1__*______
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7______*__
       8_________ */
    {
        const float sum = (tap(o, -3, -2) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 3, 2)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1______*__
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7__*______
       8_________ */
    {
        const float sum = (tap(o, -3, 2) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 3, -2)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_*_______
       3__*______
       4____0____
       5______*__
       6_______*_
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -3) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 2, 3)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_______*_
       3______*__
       4____0____
       5__*______
       6_*_______
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, 3) + tap(o, -1, 2)) + (tap(o, 0, 0) + (tap(o, 1, -2) + tap(o, 2, -3)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2________*
       3______*__
       4____0____
       5__*______
       6*________
       7_________
       8_________ */
    {
        const float sum = (tap(o, 2, -4) + tap(o, 1, -2)) + (tap(o, 0, 0) + (tap(o, -1, 2) + tap(o, -2, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2*________
       3__*______
       4____0____
       5______*__
       6________*
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -4) + tap(o, -1, -2)) + (tap(o, 0, 0) + (tap(o, 1, 2) + tap(o, 2, 4)));
        retval = fma(sum, sum, retval);
    }
    /* 0__*______
       1_________
       2___*_____
       3_________
       4____0____
       5_________
       6_____*___
       7_________
       8______*__ */
    {
        const float sum = (tap(o, -4, -2) + tap(o, -2, -1)) + (tap(o, 0, 0) + (tap(o, 2, 1) + tap(o, 4, 2)));
        retval = fma(sum, sum, retval);
    }
    /* 0______*__
       1_________
       2_____*___
       3_________
       4____0____
       5_________
       6___*_____
       7_________
       8__*______ */
    {
        const float sum = (tap(o, -4, 2) + tap(o, -2, 1)) + (tap(o, 0, 0) + (tap(o, 2, -1) + tap(o, 4, -2)));
        retval = fma(sum, sum, retval);
    }
    return retval;
}

float MaltaUnit(int o) {
    float retval;
    // x grows, y constant
    {
        const float sum = (((tap(o, 0, -4) + tap(o, 0, -3)) + (tap(o, 0, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 0, 2) + tap(o, 0, 3)))) + tap(o, 0, 4);
        retval = sum*sum;
    }
    // y grows, x constant
    {
        const float sum = (((tap(o, -4, 0) + tap(o, -3, 0)) + (tap(o, -2, 0) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 0) + tap(o, 3, 0)))) + tap(o, 4, 0);
        retval = fma(sum, sum, retval);
    }
    // both grow
    {
        const float sum = (tap(o, -3, -3) + tap(o, -2, -2)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 2, 2) + tap(o, 3, 3))));
        retval = fma(sum, sum, retval);
    }
    // y grows, x shrinks
    {
        const float sum = (tap(o, -3, 3) + tap(o, -2, 2)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 2, -2) + tap(o, 3, -3))));
        retval = fma(sum, sum, retval);
    }
    // y grows -4 to 4, x shrinks 1 -> -1
    {
        const float sum = (((tap(o, -4, 1) + tap(o, -3, 1)) + (tap(o, -2, 1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, -1) + tap(o, 3, -1)))) + tap(o, 4, -1);
        retval = fma(sum, sum, retval);
    }
    //  y grows -4 to 4, x grows -1 -> 1
    {
        const float sum = (((tap(o, -4, -1) + tap(o, -3, -1)) + (tap(o, -2, -1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 1) + tap(o, 3, 1)))) + tap(o, 4, 1);
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y grows -1 to 1
    {
        const float sum = (((tap(o, -1, -4) + tap(o, -1, -3)) + (tap(o, -1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 1, 2) + tap(o, 1, 3)))) + tap(o, 1, 4);
        retval = fma(sum, sum, retval);
    }
    // x grows -4 to 4, y shrinks 1 to -1
    {
        const float sum = (((tap(o, 1, -4) + tap(o, 1, -3)) + (tap(o, 1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, -1, 2) + tap(o, -1, 3)))) + tap(o, -1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1__*______
       2___*_____
       3___*_____
       4____0____
       5_____*___
       6_____*___
       7______*__
       8_________ */
    {
        const float sum = (tap(o, -3, -2) + tap(o, -2, -1)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 2, 1) + tap(o, 3, 2))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1______*__
       2_____*___
       3_____*___
       4____0____
       5___*_____
       6___*_____
       7__*______
       8_________ */
    {
        const float sum = (tap(o, -3, 2) + tap(o, -2, 1)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 2, -1) + tap(o, 3, -2))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_*_______
       3__**_____
       4____0____
       5_____**__
       6_______*_
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, -3) + tap(o, -1, -2)) + (tap(o, -1, -1) + ((tap(o, 0, 0) + tap(o, 1, 1)) + (tap(o, 1, 2) + tap(o, 2, 3))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_______*_
       3_____**__
       4____0____
       5__**_____
       6_*_______
       7_________
       8_________ */
    {
        const float sum = (tap(o, -2, 3) + tap(o, -1, 2)) + (tap(o, -1, 1) + ((tap(o, 0, 0) + tap(o, 1, -1)) + (tap(o, 1, -2) + tap(o, 2, -3))));
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_________
       3______***
       4___*0*___
       5***______
       6_________
       7_________
       8_________ */
    {
        const float sum = (((tap(o, 1, -4) + tap(o, 1, -3)) + (tap(o, 1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, -1, 2) + tap(o, -1, 3)))) + tap(o, -1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0_________
       1_________
       2_________
       3***______
       4___*0*___
       5______***
       6_________
       7_________
       8_________ */
    {
        const float sum = (((tap(o, -1, -4) + tap(o, -1, -3)) + (tap(o, -1, -2) + tap(o, 0, -1))) + ((tap(o, 0, 0) + tap(o, 0, 1)) + (tap(o, 1, 2) + tap(o, 1, 3)))) + tap(o, 1, 4);
        retval = fma(sum, sum, retval);
    }
    /* 0___*_____
       1___*_____
       2___*_____
       3____*____
       4____0____
       5____*____
       6_____*___
       7_____*___
       8_____*___ */
    {
        const float sum = (((tap(o, -4, -1) + tap(o, -3, -1)) + (tap(o, -2, -1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, 1) + tap(o, 3, 1)))) + tap(o, 4, 1);
        retval = fma(sum, sum, retval);
    }
    /* 0_____*___
       1_____*___
       2____ *___
       3____*____
       4____0____
       5____*____
       6___*_____
       7___*_____
       8___*_____ */
    {
        const float sum = (((tap(o, -4, 1) + tap(o, -3, 1)) + (tap(o, -2, 1) + tap(o, -1, 0))) + ((tap(o, 0, 0) + tap(o, 1, 0)) + (tap(o, 2, -1) + tap(o, 3, -1)))) + tap(o, 4, -1);
        retval = fma(sum, sum, retval);
    }
    return retval;
}


[shader("compute")]
[numthreads(16, 16, 1)]
void MaltaDiffMap(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* lum0, uniform float* lum1, uniform float* block_diff_ac, uniform uint64_t width, uniform uint64_t height, uniform float w_0gt1, uniform float w_0lt1, uniform float norm1, uniform float len, uniform float mulli) {
    const uint64_t x = threadIdx.x + blockIdx.x*16;
    const uint64_t y = threadIdx.y + blockIdx.y*16;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float w_pre0gt1 = mulli * sqrt(kWeight0 * w_0gt1) / len2;
    const float w_pre0lt1 = mulli * sqrt(kWeight1 * w_0lt1) / len2;
    const float norm2_0gt1 = w_pre0gt1 * norm1;
    const float norm2_0lt1 = w_pre0lt1 * norm1;

    int topleftx = blockIdx.x*16 - 4; int toplefty = blockIdx.y*16 - 4;

    const uint serialind = threadIdx.x + threadIdx.y * 16;
    const uint serialstride = 16*16;
    for (uint i = serialind; i < 24*24; i += serialstride){
        int workx = topleftx + i%24; int worky = toplefty + i/24;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = 0.0f;
            continue;
        }

        const float absval = 0.5f * abs(lum0[worky*width + workx]) + 0.5f * abs(lum1[worky*width + workx]);
        const float diff = lum0[worky*width + workx] - lum1[worky*width + workx];
        const float scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float scaler2 = norm2_0lt1 / ((float)(norm1) + absval);
        const float fabs0 = abs(lum0[worky*width + workx]);

        // Secondary half-open quadratic objectives.
        const float too_small = 0.55f * fabs0;
        const float too_big = 1.05f * fabs0;

        float impact = 0.0f;

        if (lum0[worky*width + workx] < 0.0f){
            if (lum1[worky*width + workx] > -too_small){
                impact = lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] < -too_big){
                impact = -lum1[worky*width + workx] - too_big;
            }
        } else {
            if (lum1[worky*width + workx] < too_small){
                impact = -lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] > too_big){
                impact = lum1[worky*width + workx] - too_big;
            }
        }
        impact *= scaler2;

        if (diff < 0.0f){
            diffs[i] -= impact;
        } else {
            diffs[i] += impact;
        }
    }
    GroupMemoryBarrierWithGroupSync(); //diffs is loaded

    //printf("local diffs : %f at %d\n", diffs[(threadIdx.y+4)*24 + threadIdx.x+4], (threadIdx.y+4)*24 + threadIdx.x+4);

    float result = MaltaUnit((int)((threadIdx.y+4)*24 + threadIdx.x+4));
    if (x < width && y < height){
        block_diff_ac[y*width + x] += result;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void MaltaDiffMapLF(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* lum0, uniform float* lum1, uniform float* block_diff_ac, uniform uint64_t width, uniform uint64_t height, uniform float w_0gt1, uniform float w_0lt1, uniform float norm1, uniform float len, uniform float mulli) {
    const uint64_t x = threadIdx.x + blockIdx.x*16;
    const uint64_t y = threadIdx.y + blockIdx.y*16;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float w_pre0gt1 = mulli * sqrt(kWeight0 * w_0gt1) / len2;
    const float w_pre0lt1 = mulli * sqrt(kWeight1 * w_0lt1) / len2;
    const float norm2_0gt1 = w_pre0gt1 * norm1;
    const float norm2_0lt1 = w_pre0lt1 * norm1;

    int topleftx = blockIdx.x*16 - 4; int toplefty = blockIdx.y*16 - 4;

    const uint serialind = threadIdx.x + threadIdx.y * 16;
    const uint serialstride = 16*16;
    for (uint i = serialind; i < 24*24; i += serialstride){
        int workx = topleftx + i%24; int worky = toplefty + i/24;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = 0.0f;
            continue;
        }

        const float absval = 0.5f * abs(lum0[worky*width + workx]) + 0.5f * abs(lum1[worky*width + workx]);
        const float diff = lum0[worky*width + workx] - lum1[worky*width + workx];
        const float scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float scaler2 = norm2_0lt1 / ((float)(norm1) + absval);
        const float fabs0 = abs(lum0[worky*width + workx]);

        // Secondary half-open quadratic objectives.
        const float too_small = 0.55f * fabs0;
        const float too_big = 1.05f * fabs0;

        float impact = 0.0f;

        if (lum0[worky*width + workx] < 0.0f){
            if (lum1[worky*width + workx] > -too_small){
                impact = lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] < -too_big){
                impact = -lum1[worky*width + workx] - too_big;
            }
        } else {
            if (lum1[worky*width + workx] < too_small){
                impact = -lum1[worky*width + workx] + too_small;
            } else if (lum1[worky*width + workx] > too_big){
                impact = lum1[worky*width + workx] - too_big;
            }
        }
        impact *= scaler2;

        if (diff < 0.0f){
            diffs[i] -= impact;
        } else {
            diffs[i] += impact;
        }
    }
    GroupMemoryBarrierWithGroupSync(); //diffs is loaded

    float result = MaltaUnitLF((int)((threadIdx.y+4)*24 + threadIdx.x+4));
    if (x < width && y < height){
        block_diff_ac[y*width + x] += result;
    }
}