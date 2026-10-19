// pycvvdp's safe_pow: (x + 1e-5)^p - 1e-5^p, clamped at 0 because the GPU's pow may round x = 0 slightly below
// the constant
float safe_pow(float x, float p){
    const float epsilon = 0.00001f;
    return max(pow(x + epsilon, p) - pow(epsilon, p), 0.f);
}

//pointer jumping, start with 1024 threads
//conditions: 1 divide by size, 2 apply the inverse power at the end, 4 apply the power to the inputs (multiplied by
//scale first), 8 use safe_pow for both powers, 16 clamp the inputs at 0.01 first
[shader("compute")]
[numthreads(1024, 1, 1)]
void reduceSum(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform int64_t size, uniform int power, uniform uint conditions, uniform float scale){
    const int64_t global_thid = threadIdx.x + blockIdx.x * 1024;
    const uint local_thid = threadIdx.x;

    static groupshared float pointerJumpingBuffer[1024]; //one float per thread

    if (global_thid >= size){
        pointerJumpingBuffer[local_thid] = 0;
    } else {
        if ((conditions&4) == 4) { //applyPower
            float value = src[global_thid]*scale;
            if ((conditions&16) == 16) value = max(value, 0.01f);
            pointerJumpingBuffer[local_thid] = ((conditions&8) == 8) ? safe_pow(value, power) : pow(value, power);
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
            res = ((conditions&8) == 8) ? safe_pow(res, 1.f/(float)power) : pow(res, 1.f/(float)power);
        }
        dst[blockIdx.x] = res;
    }
}
