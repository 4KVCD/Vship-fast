namespace butter{

//maltaUnit functions are copy pasted from google butteraugli
__device__ float2 MaltaUnitLF(float2* d, const int xs) {
    const int xs3 = 3 * xs;
    float2 retval = {0.0f, 0.0f};
    {
        // x grows, y constant
        float2 sum =
            d[-4] +
            d[-2] +
            d[0] +
            d[2] +
            d[4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float2 sum =
            d[-xs3 - xs] +
            d[-xs - xs] +
            d[0] +
            d[xs + xs] +
            d[xs3 + xs];
        retval += sum * sum;
    }
    {
        // both grow
        float2 sum =
            d[-xs3 - 3] +
            d[-xs - xs - 2] +
            d[0] +
            d[xs + xs + 2] +
            d[xs3 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float2 sum =
            d[-xs3 + 3] +
            d[-xs - xs + 2] +
            d[0] +
            d[xs + xs - 2] +
            d[xs3 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float2 sum =
            d[-xs3 - xs + 1] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 + xs - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float2 sum =
            d[-xs3 - xs - 1] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + xs + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float2 sum =
            d[-4 - xs] +
            d[-2 - xs] +
            d[0] +
            d[2 + xs] +
            d[4 + xs];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float2 sum =
            d[-4 + xs] +
            d[-2 + xs] +
            d[0] +
            d[2 - xs] +
            d[4 - xs];
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
        float2 sum =
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + 2];
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
        float2 sum =
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 - 2];
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
        float2 sum =
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[0] +
            d[xs + 2] +
            d[xs + xs + 3];
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
        float2 sum =
            d[-xs - xs + 3] +
            d[-xs + 2] +
            d[0] +
            d[xs - 2] +
            d[xs + xs - 3];
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

        float2 sum =
            d[xs + xs - 4] +
            d[xs - 2] +
            d[0] +
            d[-xs + 2] +
            d[-xs - xs + 4];
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
        float2 sum =
            d[-xs - xs - 4] +
            d[-xs - 2] +
            d[0] +
            d[xs + 2] +
            d[xs + xs + 4];
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
        float2 sum =
            d[-xs3 - xs - 2] +
            d[-xs - xs - 1] +
            d[0] +
            d[xs + xs + 1] +
            d[xs3 + xs + 2];
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
        float2 sum =
            d[-xs3 - xs + 2] +
            d[-xs - xs + 1] +
            d[0] +
            d[xs + xs - 1] +
            d[xs3 + xs - 2];
        retval += sum * sum;
    }
    return retval;
}

__device__ float2 MaltaUnit(float2* d, const int xs) {
    const int xs3 = 3 * xs;
    float2 retval = {0.0f, 0.0f};
    {
        // x grows, y constant
        float2 sum =
            d[-4] +
            d[-3] +
            d[-2] +
            d[-1] +
            d[0] +
            d[1] +
            d[2] +
            d[3] +
            d[4];
        retval += sum * sum;
    }
    {
        // y grows, x constant
        float2 sum =
            d[-xs3 - xs] +
            d[-xs3] +
            d[-xs - xs] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs] +
            d[xs3] +
            d[xs3 + xs];
        retval += sum * sum;
    }
    {
        // both grow
        float2 sum =
            d[-xs3 - 3] +
            d[-xs - xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + xs + 2] +
            d[xs3 + 3];
        retval += sum * sum;
    }
    {
        // y grows, x shrinks
        float2 sum =
            d[-xs3 + 3] +
            d[-xs - xs + 2] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs + xs - 2] +
            d[xs3 - 3];
        retval += sum * sum;
    }
    {
        // y grows -4 to 4, x shrinks 1 -> -1
        float2 sum =
            d[-xs3 - xs + 1] +
            d[-xs3 + 1] +
            d[-xs - xs + 1] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs - 1] +
            d[xs3 - 1] +
            d[xs3 + xs - 1];
        retval += sum * sum;
    }
    {
        //  y grows -4 to 4, x grows -1 -> 1
        float2 sum =
            d[-xs3 - xs - 1] +
            d[-xs3 - 1] +
            d[-xs - xs - 1] +
            d[-xs] +
            d[0] +
            d[xs] +
            d[xs + xs + 1] +
            d[xs3 + 1] +
            d[xs3 + xs + 1];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y grows -1 to 1
        float2 sum =
            d[-4 - xs] +
            d[-3 - xs] +
            d[-2 - xs] +
            d[-1] +
            d[0] +
            d[1] +
            d[2 + xs] +
            d[3 + xs] +
            d[4 + xs];
        retval += sum * sum;
    }
    {
        // x grows -4 to 4, y shrinks 1 to -1
        float2 sum =
            d[-4 + xs] +
            d[-3 + xs] +
            d[-2 + xs] +
            d[-1] +
            d[0] +
            d[1] +
            d[2 - xs] +
            d[3 - xs] +
            d[4 - xs];
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
        float2 sum =
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + xs + 1] +
            d[xs3 + 2];
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
        float2 sum =
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs + xs - 1] +
            d[xs3 - 2];
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
        float2 sum =
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[xs + 1] +
            d[xs + 2] +
            d[xs + xs + 3];
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
        float2 sum =
            d[-xs - xs + 3] +
            d[-xs + 2] +
            d[-xs + 1] +
            d[0] +
            d[xs - 1] +
            d[xs - 2] +
            d[xs + xs - 3];
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

        float2 sum =
            d[xs + xs - 4] +
            d[xs + xs - 3] +
            d[xs - 2] +
            d[xs - 1] +
            d[0] +
            d[1] +
            d[-xs + 2] +
            d[-xs + 3];
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
        float2 sum =
            d[-xs - xs - 4] +
            d[-xs - xs - 3] +
            d[-xs - 2] +
            d[-xs - 1] +
            d[0] +
            d[1] +
            d[xs + 2] +
            d[xs + 3];
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
        float2 sum =
            d[-xs3 - xs - 2] +
            d[-xs3 - 2] +
            d[-xs - xs - 1] +
            d[-xs - 1] +
            d[0] +
            d[xs] +
            d[xs + xs + 1] +
            d[xs3 + 1];
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
        float2 sum =
            d[-xs3 - xs + 2] +
            d[-xs3 + 2] +
            d[-xs - xs + 1] +
            d[-xs + 1] +
            d[0] +
            d[xs] +
            d[xs + xs - 1] +
            d[xs3 - 1];
        retval += sum * sum;
    }
    return retval;
}

__device__ float compute_impact(float lum0, float lum1, float too_small, float too_big){
    float impact;
    if (lum0 < 0.0f){
        if (lum1 > -too_small){
            impact = lum1 + too_small;
        } else if (lum1 < -too_big){
            impact = -lum1 - too_big;
        }
    } else {
        if (lum1 < too_small){
            impact = -lum1 + too_small;
        } else if (lum1 > too_big){
            impact = lum1 - too_big;
        }
    }
    return impact;
}

__global__ void MaltaDiffMap_Kernel(const float* lum0_R, const float* lum0_D, const float* lum1_R, const float* lum1_D, float* block_diff_ac0, float* block_diff_ac1, const int64_t width, const int64_t height, const float2 w_0gt1, const float2 w_0lt1, const float2 norm1, const float len, const float mulli) {
    //each block must be 16*16
    const int64_t x = threadIdx.x + 2*blockIdx.x*blockDim.x;
    const int64_t y = threadIdx.y + 2*blockIdx.y*blockDim.y;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float2 w_pre0gt1 = sqrtf(w_0gt1 * kWeight0) * mulli / len2;
    const float2 w_pre0lt1 = sqrtf(w_0lt1 * kWeight1) * mulli / len2;
    const float2 norm2_0gt1 = w_pre0gt1 * norm1;
    const float2 norm2_0lt1 = w_pre0lt1 * norm1;

    //this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
    __shared__ float2 diffs[40*40]; //4 around each border. 40*40 = 1.25² * 32*32 -> around 1.56 values per thread. Note that we can pad in advance with negligible register cost
    //shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)
    int topleftx = blockIdx.x*blockDim.x - 4; int toplefty = blockIdx.y*blockDim.y - 4;

    const int64_t serialind = threadIdx.x + threadIdx.y * blockDim.x;
    const int64_t serialstride = blockDim.x*blockDim.y;
    for (int64_t i = serialind; i < 40*40; i += serialstride){
        int workx = topleftx + i%40; int worky = toplefty + i/40;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = {0.0f, 0.0f};
            continue;
        }

        const float2 lum0 = {lum0_R[worky*width + workx], lum1_R[worky*width + workx]};
        const float2 lum1 = {lum0_D[worky*width + workx], lum1_D[worky*width + workx]};

        const float2 absval = abs(lum0) * 0.5f + abs(lum1) * 0.5f;
        const float2 diff = lum0 - lum1;
        const float2 scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float2 scaler2 = norm2_0lt1 / (norm1 + absval);
        const float2 fabs0 = abs(lum0);

        // Secondary half-open quadratic objectives.
        const float2 too_small = fabs0 * 0.55f;
        const float2 too_big = fabs0 * 1.05f;

        float2 impact;
        impact.x = compute_impact(lum0.x, lum1.x, too_small.x, too_big.x);
        impact.y = compute_impact(lum0.y, lum1.y, too_small.y, too_big.y);

        impact = impact * scaler2;

        if (diff.x < 0.0f){
            diffs[i].x -= impact.x;
        } else {
            diffs[i].x += impact.x;
        }

        if (diff.y < 0.0f){
            diffs[i].y -= impact.y;
        } else {
            diffs[i].y += impact.y;
        }
    }
    __syncthreads(); //diffs is loaded

    //printf("local diffs : %f at %d\n", diffs[(threadIdx.y+4)*24 + threadIdx.x+4], (threadIdx.y+4)*24 + threadIdx.x+4);
    #pragma unroll
    for (int tile_x = 0; tile_x < 32; tile_x+=16){
        #pragma unroll
        for (int tile_y = 0; tile_y < 32; tile_y+=16){
            const int64_t local_x = x + tile_x;
            const int64_t local_y = y + tile_y;
            float2 result = MaltaUnit(diffs+(threadIdx.y+4+tile_y)*40 + threadIdx.x+4+tile_x, 40);
            if (local_x < width && local_y < height){
                block_diff_ac0[local_y*width + local_x] += result.x;
                block_diff_ac1[local_y*width + local_x] += result.y;
            }
        }
    }
}

__global__ void MaltaDiffMapLF_Kernel(const float* lum0_R, const float* lum0_D, const float* lum1_R, const float* lum1_D, float* block_diff_ac0, float* block_diff_ac1, const int64_t width, const int64_t height, const float2 w_0gt1, const float2 w_0lt1, const float2 norm1, const float len, const float mulli) {
    //each block must be 16*16
    const int64_t x = threadIdx.x + 2*blockIdx.x*blockDim.x;
    const int64_t y = threadIdx.y + 2*blockIdx.y*blockDim.y;

    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;

    const float len2 = len * 2.0f + 1.0f;
    const float2 w_pre0gt1 = sqrtf(w_0gt1 * kWeight0) * mulli / len2;
    const float2 w_pre0lt1 = sqrtf(w_0lt1 * kWeight1) * mulli / len2;
    const float2 norm2_0gt1 = w_pre0gt1 * norm1;
    const float2 norm2_0lt1 = w_pre0lt1 * norm1;

    //this is were the fancy idea start about storing diff to avoid creating a padded array for each thread (which consumes too much registers)
    __shared__ float2 diffs[40*40]; //4 around each border. 40*40 = 1.25² * 32*32 -> around 1.56 values per thread. Note that we can pad in advance with negligible register cost
    //shared memory cost: 24*24 = 576 -> 2kB (very small in front of 16kB)
    int topleftx = 2*blockIdx.x*blockDim.x - 4; int toplefty = 2*blockIdx.y*blockDim.y - 4;

    const int64_t serialind = threadIdx.x + threadIdx.y * blockDim.x;
    const int64_t serialstride = blockDim.x*blockDim.y;
    for (int64_t i = serialind; i < 40*40; i += serialstride){
        int workx = topleftx + i%40; int worky = toplefty + i/40;
        if (workx < 0 || workx >= width || worky < 0 || worky >= height){
            diffs[i] = {0.0f, 0.0f};
            continue;
        }

        const float2 lum0 = {lum0_R[worky*width + workx], lum1_R[worky*width + workx]};
        const float2 lum1 = {lum0_D[worky*width + workx], lum1_D[worky*width + workx]};

        const float2 absval = abs(lum0) * 0.5f + abs(lum1) * 0.5f;
        const float2 diff = lum0 - lum1;
        const float2 scaler = norm2_0gt1 / (norm1 + absval);

        // Primary symmetric quadratic objective.
        diffs[i] = scaler * diff;

        const float2 scaler2 = norm2_0lt1 / (norm1 + absval);
        const float2 fabs0 = abs(lum0);

        // Secondary half-open quadratic objectives.
        const float2 too_small = fabs0 * 0.55f;
        const float2 too_big = fabs0 * 1.05f;

        float2 impact;
        impact.x = compute_impact(lum0.x, lum1.x, too_small.x, too_big.x);
        impact.y = compute_impact(lum0.y, lum1.y, too_small.y, too_big.y);

        impact = impact * scaler2;

        if (diff.x < 0.0f){
            diffs[i].x -= impact.x;
        } else {
            diffs[i].x += impact.x;
        }

        if (diff.y < 0.0f){
            diffs[i].y -= impact.y;
        } else {
            diffs[i].y += impact.y;
        }
    }
    __syncthreads(); //diffs is loaded

    //printf("local diffs : %f at %d\n", diffs[(threadIdx.y+4)*24 + threadIdx.x+4], (threadIdx.y+4)*24 + threadIdx.x+4);
    #pragma unroll
    for (int tile_x = 0; tile_x < 32; tile_x+=16){
        #pragma unroll
        for (int tile_y = 0; tile_y < 32; tile_y+=16){
            const int64_t local_x = x + tile_x;
            const int64_t local_y = y + tile_y;
            float2 result = MaltaUnitLF(diffs+(threadIdx.y+4+tile_y)*40 + threadIdx.x+4+tile_x, 40);
            if (local_x < width && local_y < height){
                block_diff_ac0[local_y*width + local_x] += result.x;
                block_diff_ac1[local_y*width + local_x] += result.y;
            }
        }
    }
}

__host__ void MaltaDiffMap(const float* lum0_R, const float* lum0_D, const float* lum1_R, const float* lum1_D, float* block_diff_ac0, float* block_diff_ac1, const int64_t width, const int64_t height, const float2 w_0gt1, const float2 w_0lt1, const float2 norm1, hipStream_t stream){
    const float len = 3.75f;
    const float mulli = 0.39905817637f;

    const int64_t th_x = 16;
    const int64_t th_y = 16;
    const int64_t bl_x = (width - 1)/(th_x*2) + 1;
    const int64_t bl_y = (height - 1)/(th_y*2) + 1;
    MaltaDiffMap_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(lum0_R, lum0_D, lum1_R, lum1_D, block_diff_ac0, block_diff_ac1, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
    GPU_CHECK(hipGetLastError());
}

__host__ void MaltaDiffMapLF(const float* lum0_R, const float* lum0_D, const float* lum1_R, const float* lum1_D, float* block_diff_ac0, float* block_diff_ac1, const int64_t width, const int64_t height, const float2 w_0gt1, const float2 w_0lt1, const float2 norm1, hipStream_t stream){
    const float len = 3.75f;
    const float mulli = 0.611612573796f;

    const int64_t th_x = 16;
    const int64_t th_y = 16;
    const int64_t bl_x = (width - 1)/(th_x*2) + 1;
    const int64_t bl_y = (height - 1)/(th_y*2) + 1;
    MaltaDiffMapLF_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(lum0_R, lum0_D, lum1_R, lum1_D, block_diff_ac0, block_diff_ac1, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
    GPU_CHECK(hipGetLastError());
}

}
