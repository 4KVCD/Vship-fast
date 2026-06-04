//pointer jumping, start with 1024 threads
[shader("compute")]
[numthreads(1024, 1, 1)]
void reduceSum(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t size, uniform int power, uniform uint conditions){
    const int64_t global_thid = threadIdx.x + blockIdx.x * 1024;
    const uint local_thid = threadIdx.x;

    static groupshared float pointerJumpingBuffer[1024]; //one float per thread

    if (global_thid >= size){
        pointerJumpingBuffer[local_thid] = 0;
    } else {
        if ((conditions&4) == 4) { //applyPower
            pointerJumpingBuffer[local_thid] = pow(src[global_thid], power);
        } else {
            pointerJumpingBuffer[local_thid] = src[global_thid];
        }
    }

    GroupMemoryBarrierWithGroupSync();

    int next = 1024;
    while (next > 1){
        next /= 2;
        if (local_thid < next){
            pointerJumpingBuffer[local_thid] += pointerJumpingBuffer[local_thid+next];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_thid == 0){
        float res = pointerJumpingBuffer[0];
        if ((conditions&1) == 1) res /= size; //divide
        if ((conditions&2) == 2) { //applyInversePower
            res = pow(res, 1.f/(float)power);
        }
        dst[blockIdx.x] = res;
    }
}
