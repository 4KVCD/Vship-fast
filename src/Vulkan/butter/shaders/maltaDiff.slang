
//this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
groupshared float diffs[24*24]; //4 around each border. 24*24 = 1.5² * 16*16 -> around 2.25 values per thread. Note that we can pad in advance with negligible register cost
//shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)

//maltaUnit functions are copy pasted from google butteraugli
float MaltaUnitLF(uint64_t offset) {
    float retval = 0.0f;
    {
        // x grows, y constant
        float sum =
            diffs[offset-4] +
            diffs[offset-2] +
            diffs[offset+0] +
            diffs[offset+2] +
            diffs[offset+4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float sum =
            diffs[offset-72 - 24] +
            diffs[offset-24 - 24] +
            diffs[offset+0] +
            diffs[offset+24 + 24] +
            diffs[offset+72 + 24];
        retval += sum * sum;
    }
    {
        // both grow
        float sum =
            diffs[offset-72 - 3] +
            diffs[offset-24 - 24 - 2] +
            diffs[offset+0] +
            diffs[offset+24 + 24 + 2] +
            diffs[offset+72 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float sum =
            diffs[offset-72 + 3] +
            diffs[offset-24 - 24 + 2] +
            diffs[offset+0] +
            diffs[offset+24 + 24 - 2] +
            diffs[offset+72 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float sum =
            diffs[offset-72 - 24 + 1] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 + 24 - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float sum =
            diffs[offset-72 - 24 - 1] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 24 + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float sum =
            diffs[offset-4 - 24] +
            diffs[offset-2 - 24] +
            diffs[offset+0] +
            diffs[offset+2 + 24] +
            diffs[offset+4 + 24];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float sum =
            diffs[offset-4 + 24] +
            diffs[offset-2 + 24] +
            diffs[offset+0] +
            diffs[offset+2 - 24] +
            diffs[offset+4 - 24];
        retval += sum * sum;
    }
    {
        /* 0_________
        1__*______
        2___*_____
        3_________
        4____0____
        5_________
        6_____*___
        7______*__
        8_________ */
        float sum =
            diffs[offset-72 - 2] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1______*__
        2_____*___
        3_________
        4____0____
        5_________
        6___*_____
        7__*______
        8_________ */
        float sum =
            diffs[offset-72 + 2] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 - 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_*_______
        3__*______
        4____0____
        5______*__
        6_______*_
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 - 3] +
            diffs[offset-24 - 2] +
            diffs[offset+0] +
            diffs[offset+24 + 2] +
            diffs[offset+24 + 24 + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_______*_
        3______*__
        4____0____
        5__*______
        6_*_______
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 + 3] +
            diffs[offset-24 + 2] +
            diffs[offset+0] +
            diffs[offset+24 - 2] +
            diffs[offset+24 + 24 - 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2________*
        3______*__
        4____0____
        5__*______
        6*________
        7_________
        8_________ */

        float sum =
            diffs[offset+24 + 24 - 4] +
            diffs[offset+24 - 2] +
            diffs[offset+0] +
            diffs[offset-24 + 2] +
            diffs[offset-24 - 24 + 4];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2*________
        3__*______
        4____0____
        5______*__
        6________*
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 - 4] +
            diffs[offset-24 - 2] +
            diffs[offset+0] +
            diffs[offset+24 + 2] +
            diffs[offset+24 + 24 + 4];
        retval += sum * sum;
    }
    {
        /* 0__*______
        1_________
        2___*_____
        3_________
        4____0____
        5_________
        6_____*___
        7_________
        8______*__ */
        float sum =
            diffs[offset-72 - 24 - 2] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 24 + 2];
        retval += sum * sum;
    }
    {
        /* 0______*__
        1_________
        2_____*___
        3_________
        4____0____
        5_________
        6___*_____
        7_________
        8__*______ */
        float sum =
            diffs[offset-72 - 24 + 2] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 + 24 - 2];
        retval += sum * sum;
    }
    return retval;
}

float MaltaUnit(uint64_t offset) {
    float retval = 0.0f;
    {
        // x grows, y constant
        float sum =
            diffs[offset-4] +
            diffs[offset-3] +
            diffs[offset-2] +
            diffs[offset-1] +
            diffs[offset+0] +
            diffs[offset+1] +
            diffs[offset+2] +
            diffs[offset+3] +
            diffs[offset+4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float sum =
            diffs[offset-72 - 24] +
            diffs[offset-72] +
            diffs[offset-24 - 24] +
            diffs[offset-24] +
            diffs[offset+0] +
            diffs[offset+24] +
            diffs[offset+24 + 24] +
            diffs[offset+72] +
            diffs[offset+72 + 24];
        retval += sum * sum;
    }
    {
        // both grow
        float sum =
            diffs[offset-72 - 3] +
            diffs[offset-24 - 24 - 2] +
            diffs[offset-24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 1] +
            diffs[offset+24 + 24 + 2] +
            diffs[offset+72 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float sum =
            diffs[offset-72 + 3] +
            diffs[offset-24 - 24 + 2] +
            diffs[offset-24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 - 1] +
            diffs[offset+24 + 24 - 2] +
            diffs[offset+72 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float sum =
            diffs[offset-72 - 24 + 1] +
            diffs[offset-72 + 1] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset-24] +
            diffs[offset+0] +
            diffs[offset+24] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 - 1] +
            diffs[offset+72 + 24 - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float sum =
            diffs[offset-72 - 24 - 1] +
            diffs[offset-72 - 1] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset-24] +
            diffs[offset+0] +
            diffs[offset+24] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 1] +
            diffs[offset+72 + 24 + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float sum =
            diffs[offset-4 - 24] +
            diffs[offset-3 - 24] +
            diffs[offset-2 - 24] +
            diffs[offset-1] +
            diffs[offset+0] +
            diffs[offset+1] +
            diffs[offset+2 + 24] +
            diffs[offset+3 + 24] +
            diffs[offset+4 + 24];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float sum =
            diffs[offset-4 + 24] +
            diffs[offset-3 + 24] +
            diffs[offset-2 + 24] +
            diffs[offset-1] +
            diffs[offset+0] +
            diffs[offset+1] +
            diffs[offset+2 - 24] +
            diffs[offset+3 - 24] +
            diffs[offset+4 - 24];
        retval += sum * sum;
    }
    {
        /* 0_________
        1__*______
        2___*_____
        3___*_____
        4____0____
        5_____*___
        6_____*___
        7______*__
        8_________ */
        float sum =
            diffs[offset-72 - 2] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset-24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 1] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1______*__
        2_____*___
        3_____*___
        4____0____
        5___*_____
        6___*_____
        7__*______
        8_________ */
        float sum =
            diffs[offset-72 + 2] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset-24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 - 1] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 - 2];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_*_______
        3__**_____
        4____0____
        5_____**__
        6_______*_
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 - 3] +
            diffs[offset-24 - 2] +
            diffs[offset-24 - 1] +
            diffs[offset+0] +
            diffs[offset+24 + 1] +
            diffs[offset+24 + 2] +
            diffs[offset+24 + 24 + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_______*_
        3_____**__
        4____0____
        5__**_____
        6_*_______
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 + 3] +
            diffs[offset-24 + 2] +
            diffs[offset-24 + 1] +
            diffs[offset+0] +
            diffs[offset+24 - 1] +
            diffs[offset+24 - 2] +
            diffs[offset+24 + 24 - 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2_________
        3______**_
        4____0*___
        5__**_____
        6**_______
        7_________
        8_________ */

        float sum =
            diffs[offset+24 + 24 - 4] +
            diffs[offset+24 + 24 - 3] +
            diffs[offset+24 - 2] +
            diffs[offset+24 - 1] +
            diffs[offset+0] +
            diffs[offset+1] +
            diffs[offset-24 + 2] +
            diffs[offset-24 + 3];
        retval += sum * sum;
    }
    {
        /* 0_________
        1_________
        2**_______
        3__**_____
        4____0*___
        5______**_
        6_________
        7_________
        8_________ */
        float sum =
            diffs[offset-24 - 24 - 4] +
            diffs[offset-24 - 24 - 3] +
            diffs[offset-24 - 2] +
            diffs[offset-24 - 1] +
            diffs[offset+0] +
            diffs[offset+1] +
            diffs[offset+24 + 2] +
            diffs[offset+24 + 3];
        retval += sum * sum;
    }
    {
        /* 0__*______
        1__*______
        2___*_____
        3___*_____
        4____0____
        5____*____
        6_____*___
        7_____*___
        8_________ */
        float sum =
            diffs[offset-72 - 24 - 2] +
            diffs[offset-72 - 2] +
            diffs[offset-24 - 24 - 1] +
            diffs[offset-24 - 1] +
            diffs[offset+0] +
            diffs[offset+24] +
            diffs[offset+24 + 24 + 1] +
            diffs[offset+72 + 1];
        retval += sum * sum;
    }
    {
        /* 0______*__
        1______*__
        2_____*___
        3_____*___
        4____0____
        5____*____
        6___*_____
        7___*_____
        8_________ */
        float sum =
            diffs[offset-72 - 24 + 2] +
            diffs[offset-72 + 2] +
            diffs[offset-24 - 24 + 1] +
            diffs[offset-24 + 1] +
            diffs[offset+0] +
            diffs[offset+24] +
            diffs[offset+24 + 24 - 1] +
            diffs[offset+72 - 1];
        retval += sum * sum;
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

    float result = MaltaUnit((threadIdx.y+4)*24 + threadIdx.x+4);
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

    float result = MaltaUnitLF((threadIdx.y+4)*24 + threadIdx.x+4);
    if (x < width && y < height){
        block_diff_ac[y*width + x] += result;
    }
}