float* getFramePointer(int i, float* temporalBuffer, int64_t width, int64_t height, int temporal_offset, int temporal_size, int max_temporal_size){
    i = min(temporal_size-1, i);
    return temporalBuffer + ((temporal_offset+i)%max_temporal_size)*(3*width*height);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void temporalConvolutionKernel_d(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* Y_sustained, uniform float* RG_sustained, uniform float* YV_sustained, uniform float* Y_transient, uniform int64_t width, uniform int64_t height, uniform float* temporalkernel, uniform float* temporalBuffer, uniform int temporal_offset, uniform int temporal_size, uniform int max_temporal_size){
    const int64_t x = threadIdx.x + blockIdx.x * 256;
    const int64_t planeSize = width*height;
    if (x >= planeSize) return;

    if (temporal_size == 1){
        float* srcptr = getFramePointer(0, temporalBuffer, width, height, temporal_offset, temporal_size, max_temporal_size);
        Y_sustained[x] = srcptr[x];
        RG_sustained[x] = srcptr[x+planeSize];
        YV_sustained[x] = srcptr[x+2*planeSize];
        Y_transient[x] = 0.f;
        //if (x == 13*1024 + 64) printf("after temporalFilter: %f %f %f %f\n", Y_sustained[x], RG_sustained[x], YV_sustained[x], Y_transient[x]);
        return;
    }

    //we use float2 to use double issue float ALU computing units.
    float2 valueTemp;

    float2 kernelTemp;

    float2 resY_Y = {0.f, 0.f};
    float2 resRG_YV = {0.f, 0.f};
    //convolution over temporal dimension
    for (int k = 0;  k < max_temporal_size; k++){
        //kth frame
        float* srcptr = getFramePointer(k, temporalBuffer, width, height, temporal_offset, temporal_size, max_temporal_size);

        kernelTemp.x = temporalkernel[k];
        kernelTemp.y = temporalkernel[k+3*max_temporal_size];
        valueTemp.x = srcptr[x];
        valueTemp.y = valueTemp.x;

        resY_Y = fma(valueTemp, kernelTemp, resY_Y);

        kernelTemp.x = temporalkernel[k+max_temporal_size];
        kernelTemp.y = temporalkernel[k+2*max_temporal_size];
        valueTemp.x = srcptr[x + planeSize];
        valueTemp.y = srcptr[x + planeSize*2];

        resRG_YV = fma(valueTemp, kernelTemp, resRG_YV);
    }
    Y_sustained[x] = resY_Y.x;
    Y_transient[x] = resY_Y.y;
    RG_sustained[x] = resRG_YV.x;
    YV_sustained[x] = resRG_YV.y;

    //if (x == 13*1024 + 64) printf("after temporalFilter: %f %f %f %f\n", resY_Y.x, resRG_YV.x, resRG_YV.y, resY_Y.y);
}