//import gaussianBlurSSIMU2;

groupshared float sum[6][1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void sumreduce(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* dst, uniform float* src, uniform uint64_t widthPerElement, uniform uint64_t bl_x){
    //dst must be of size 6*blocknum at least
    const int64_t x = threadIdx.x + blockIdx.x*1024;
    const int64_t thx = threadIdx.x;
    const int64_t threadnum = 1024;
    
    if (x >= widthPerElement){
        
        for (int i = 0; i < 6; i++){
            sum[i][thx] = 0;
        }
    } else {
        
        for (int i = 0; i < 6; i++){
            sum[i][thx] = src[x + i*widthPerElement];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            
            for (int i = 0; i < 6; i++){
                sum[i][thx] += sum[i][thx+next];
            }
        }
        next *= 2;
        GroupMemoryBarrierWithGroupSync();
    }

    if (thx == 0){
        
        for (int i = 0; i < 6; i++){
            dst[i*bl_x + blockIdx.x] = sum[i][0];
        }
    }
}