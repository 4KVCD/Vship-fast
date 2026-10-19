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

// One reduceSum dispatch of a batch (ReduceJob in pooling.hpp)
struct ReduceJob {
    float* dst;
    float* src;
    int64_t size;
    uint conditions;
    float scale;
};

// The value input i of a job contributes to its sum, as in reduceSum
float reduceInput(ReduceJob job, int64_t i, int power){
    if (i >= job.size) return 0;
    if ((job.conditions&4) == 4){ //applyPower
        float value = job.src[i]*job.scale;
        if ((job.conditions&16) == 16) value = max(value, 0.01f);
        return ((job.conditions&8) == 8) ? safe_pow(value, power) : pow(value, power);
    }
    return job.src[i];
}

// reduceSum for several jobs at once (blockIdx.y), with the same sums in the same order. reduceSum adds the 1024
// inputs of a workgroup in a tree: input i and i+512, then i and i+256 of those, and so on. Here a team of 32
// threads takes each block of 1024: thread t adds inputs t + 32k (k < 32) in the tree's first 5 steps in registers,
// and the team's first thread adds the 32 results in its last 5 steps. No barrier between the steps, and every
// job of a pass in one dispatch.
static const int TEAM = 32;   // threads per block of 1024 inputs
static const int TEAMS = 8;   // blocks per workgroup
groupshared float teamSums[TEAMS][TEAM];

[shader("compute")]
[numthreads(TEAM, TEAMS, 1)]
void reduceSums(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform ReduceJob* jobs, uniform int power){
    const ReduceJob job = jobs[blockIdx.y];
    const int64_t block = (int64_t)blockIdx.x*TEAMS + threadIdx.y;
    const bool active = block < (job.size + 1023)/1024; // the dispatch has the workgroups of its largest job
    const int lane = threadIdx.x;

    float inputs[32];
    [ForceUnroll]
    for (int k = 0; k < 32; k++) inputs[k] = active ? reduceInput(job, block*1024 + lane + TEAM*k, power) : 0.f;
    [ForceUnroll]
    for (int level = 4; level >= 0; level--){
        const int step = 1 << level;
        [ForceUnroll]
        for (int k = 0; k < 16; k++) if (k < step) inputs[k] += inputs[k + step];
    }
    teamSums[threadIdx.y][lane] = inputs[0];
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0 && active){
        float sums[TEAM];
        [ForceUnroll]
        for (int l = 0; l < TEAM; l++) sums[l] = teamSums[threadIdx.y][l];
        [ForceUnroll]
        for (int level = 4; level >= 0; level--){
        const int step = 1 << level;
            [ForceUnroll]
            for (int l = 0; l < 16; l++) if (l < step) sums[l] += sums[l + step];
        }
        float res = sums[0];
        if ((job.conditions&1) == 1) res /= job.size; //divide
        if ((job.conditions&2) == 2) { //applyInversePower
            res = ((job.conditions&8) == 8) ? safe_pow(res, 1.f/(float)power) : pow(res, 1.f/(float)power);
        }
        job.dst[block] = res;
    }
}
