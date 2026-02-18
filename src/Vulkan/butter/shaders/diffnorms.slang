
groupshared float sharedmem[1024*3];

[shader("compute")]
[numthreads(1024, 1, 1)]
void sumreduce(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t bl_x){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is 3*sizeof(float)*threadnum at least
    const uint64_t x = threadIdx.x + blockIdx.x*1024;
    const uint64_t thx = threadIdx.x;
    const uint64_t threadnum = 1024;

    if (x >= width){
        sharedmem[thx] = 0.0f;
        sharedmem[1024+thx] = 0.0f;
        sharedmem[1024*2+thx] = 0.0f;
    } else {
        sharedmem[thx] = src[x];
        sharedmem[1024+thx] = src[x+width];
        sharedmem[1024*2+thx] = src[x+2*width];
    }
    GroupMemoryBarrierWithGroupSync();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] += sharedmem[thx+next];
            sharedmem[thx + 1024] += sharedmem[thx+next + 1024];
            sharedmem[thx + 2*1024] = max(sharedmem[thx + 2*1024], sharedmem[thx+next + 2*1024]);
        }
        next *= 2;
        GroupMemoryBarrierWithGroupSync();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0];
        dst[blockIdx.x + bl_x] = sharedmem[1024];
        dst[blockIdx.x + 2*bl_x] = sharedmem[2*1024];
    }
}

[shader("compute")]
[numthreads(1024, 1, 1)]
void sumreducenorm(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t width, uniform uint64_t bl_x, uniform uint Qnorm){
    //dst must be of size 3*sizeof(float)*blocknum at least
    //shared memory needed is sizeof(float)*threadnum at least
    const uint64_t x = threadIdx.x + blockIdx.x*1024;
    const uint64_t thx = threadIdx.x;
    const uint64_t threadnum = 1024;

    if (x >= width){
        sharedmem[thx] = 0.0f;
        sharedmem[1024+thx] = 0.0f;
        sharedmem[1024*2+thx] = 0.0f;
    } else {
        //instead of norm2 we will have Qnorm now!
        sharedmem[thx] = pow(src[x], Qnorm);
        sharedmem[1024+thx] = abs(src[x])*src[x]*src[x];
        sharedmem[1024*2+thx] = abs(src[x]);
    }
    GroupMemoryBarrierWithGroupSync();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] += sharedmem[thx+next];
            sharedmem[thx + 1024] += sharedmem[thx+next + 1024];
            sharedmem[thx + 2*1024] = max(sharedmem[thx + 2*1024], sharedmem[thx+next + 2*1024]);
        }
        next *= 2;
        GroupMemoryBarrierWithGroupSync();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0]/width;
        dst[blockIdx.x + bl_x] = sharedmem[1024]/width;
        dst[blockIdx.x + 2*bl_x] = sharedmem[2*1024];
    }
}