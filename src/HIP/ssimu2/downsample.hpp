#pragma once

namespace ssimu2{

#include "makeXYB.hpp"

struct downsampleKernelInformation{
    bool result_XYB;
    bool skipX; //skip writing of X source scale to XYB
    bool skipY;
    bool skipB;
};

__launch_bounds__(256)
__global__ void downsamplekernel(float* src1, float* src2, float* src3, float* dst1, float* dst2, float* dst3, int64_t width, int64_t height, downsampleKernelInformation info){ //threads represents output pixels
    int64_t x = threadIdx.x + blockIdx.x*blockDim.x; // < width >> 1 +1
    int64_t y = threadIdx.y + blockIdx.y*blockDim.y; // < height >> 1 +1

    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    if (x >= neww || y >= newh) return;

    float3 srcData[4];
    #pragma unroll
    for (int dy = 0; dy < 2; dy++){
        #pragma unroll
        for (int dx = 0; dx < 2; dx++){
            srcData[dy*2+dx].x = src1[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].y = src2[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
            srcData[dy*2+dx].z = src3[min((int)(2*y + dy), (int)(height-1)) * width + min((int)(2*x+dx), (int)(width-1))];
        }
    }

    float3 dst = {0.f, 0.f, 0.f};
    #pragma unroll
    for (int i = 0; i < 4; i++){
        dst += srcData[i];
        //to later write srcData in XYB
        rgb_to_positive_xyb_d(srcData[i]);
    }
    dst *= 0.25f;

    //done on last scale since the score won't be used for further downsampling
    if (info.result_XYB){
        rgb_to_positive_xyb_d(dst);
    }

    dst1[y * neww + x] = dst.x;
    dst2[y * neww + x] = dst.y;
    dst3[y * neww + x] = dst.z;

    //write on source the XYB version directly with in registers data to reduce memory reads
    #pragma unroll
    for (int dy = 0; dy < 2; dy++){
        #pragma unroll
        for (int dx = 0; dx < 2; dx++){
            if (2*y+dy < height && 2*x+dx < width){
                if (!info.skipX) src1[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].x;
                if (!info.skipY) src2[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].y;
                if (!info.skipB) src3[(2*y+dy)*width+2*x+dx] = srcData[dy*2+dx].z;
            }
        }
    }
}

void inline downsample(float* src1, float* src2, float* src3, float* dst1, float* dst2, float* dst3, int64_t width, int64_t height, downsampleKernelInformation info, hipStream_t stream){
    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    int64_t th_x = std::min((int64_t)16, neww);
    int64_t th_y = std::min((int64_t)16, newh);
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;
    downsamplekernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(src1, src2, src3, dst1, dst2, dst3, width, height, info);
    GPU_CHECK(hipGetLastError());
}

}
