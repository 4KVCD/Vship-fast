#pragma once

#include "csf.hpp"

namespace cvvdp{

__device__ constexpr float kernel_a = 0.4f;
__device__ constexpr float gaussPyrKernel[5] = {0.25-kernel_a/2.,0.25, kernel_a, 0.25, 0.25-kernel_a/2.};

//use block of size 16/16
//pyramidSize is the stride of a "big" plane as a whole. mem_d needs to be offset prior using scaleOffset
//memo: mem_d+scaleOffset+pyramidSize*n = mem_d+pyramidSize*n + scaleOffset (-> we only need mem_d+scaleOffset and not mem_d)
//this kernel is called for all planes and then all scales successively except the smallest one
//both sources are treated at once
__global__ void fromPyrReduceToCSF_Kernel(float* mem_d, float* mem_d2, const int64_t pyramidSize, const int64_t width, const int64_t height, const int scale, const int channel, CSF_Handler csfhandle){
    const int64_t base_x = blockIdx.x * 16;
    const int64_t base_y = blockIdx.y * 16;
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int thid = thy*16+thx;

    const int64_t small_width = (width+1)/2;
    const int64_t small_height = (height+1)/2;

    //memory layout helpers
    float* inputPlane1 = mem_d+channel*pyramidSize; //read from
    float* inputPlane2 = mem_d2+channel*pyramidSize; //read from
    float* Lbkg1 = mem_d+4*pyramidSize; //gets written by channel 0, read by others
    float* Lbkg2 = mem_d2+4*pyramidSize; //gets written by channel 0, read by others
    float* smallScale1 = inputPlane1+width*height; //gets written to
    float* smallScale2 = inputPlane2+width*height; //gets written to
    float* outputPlane1 = mem_d+(channel+5)*pyramidSize; //gets written to
    float* outputPlane2 = mem_d2+(channel+5)*pyramidSize; //gets written to
    __shared__ float2 sharedmem[23*23 + 10*10];
    float2* bufferInput = sharedmem; // 24*24 (16*16 padded by 4 all side)
    float2* bufferSmallScale = sharedmem+23*23; //10*10 (8*8 padded by 1 all side)

    //load input in memory
    #pragma unroll
    for (int i = thid; i < 23*23; i+=256){
        int64_t x = base_x-4 + (i%23);
        int64_t y = base_y-4 + (i/23);

        while (x < 0 || x >= width){
            if (x < 0) x = -x-1;
            if (x >= width) x = 2*width - x - 2;
        }

        while (y < 0 || y >= height){
            if (y < 0) y = -y-1;
            if (y >= height) y = 2*height - y - 2;
        }

        bufferInput[i].x = inputPlane1[y*width+x];
        bufferInput[i].y = inputPlane2[y*width+x];
    }
    __syncthreads();

    //compute into bufferSmallScale
    if (thx < 10 && thy < 10){
        bufferSmallScale[thy*10+thx] = {0.f, 0.f};
        #pragma unroll
        for (int ix = 0; ix < 5; ix++){
            const float kernelx = gaussPyrKernel[ix];
            #pragma unroll
            for (int iy = 0; iy < 5; iy++){
                const float kernel = kernelx*gaussPyrKernel[iy];
                //for thx = 9, base column is 18
                bufferSmallScale[thy*10+thx] += bufferInput[(2*thy+iy)*23+2*thx+ix]*kernel;
            }
        }
    }
    __syncthreads();
    //export [1x1, 8x8] smallscale into plane
    if (thx < 8 && thy < 8){
        const int64_t new_x = (blockIdx.x*8 + thx);
        const int64_t new_y = (blockIdx.y*8 + thy);
        //if (new_x == 4 && new_y == 7) printf("got %f in channel %d band %d\n", bufferSmallScale[(thy+1)*10+thx+1], channel, scale);
        if (new_x < small_width && new_y < small_height){
            smallScale1[new_y*small_width+new_x] = bufferSmallScale[(thy+1)*10+thx+1].x;
            smallScale2[new_y*small_width+new_x] = bufferSmallScale[(thy+1)*10+thx+1].y;
        }
    }
    //no need to syncthreads because bufferSmallScale won't be written to again

    //use [0x0, 9x9] smallscale to expand and get local value for global x global y
    const int64_t glx = base_x + thx;
    const int64_t gly = base_y + thy;
    float2 expanded = {0.f, 0.f};

    const int parity_x = thx%2;
    const int parity_y = thy%2;

    //either in [0,3] or [0, 2]
    for (int dx = 0; dx < 3-parity_x; dx++){
        const float kernelx = 2*gaussPyrKernel[2*dx+parity_x];
        for (int dy = 0; dy < 3-parity_y; dy++){
            const float kernel = kernelx*2*gaussPyrKernel[2*dy+parity_y];
            expanded += bufferSmallScale[((thy+1)/2 + dy) * 10 + ((thx+1)/2 + dx)]*kernel;
        }
    }

    //clamp + export this value as Lbkg if channel = 0 or read otherwise
    float2 lbkg = {0.f, 0.f};
    if (channel == 0){
        expanded = max(expanded, 0.01f);
        lbkg = expanded;
        if (glx < width && gly < height) {
            Lbkg1[gly*width+glx] = lbkg.x;
            Lbkg2[gly*width+glx] = lbkg.y;
        }
    } else {
        int64_t ref_ind_x = glx;
        int64_t ref_ind_y = gly;

        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= width) ref_ind_x = 2*width - ref_ind_x - 2;

        if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
        if (ref_ind_y >= height) ref_ind_y = 2*height - ref_ind_y - 2;

        lbkg.x = Lbkg1[ref_ind_y*width+ref_ind_x];
        lbkg.y = Lbkg2[ref_ind_y*width+ref_ind_x];
    }

    //substract + refine to get final contrast on this local value
    const float2 Ldiff = (bufferInput[(thy+4)*23 + thx+4] - expanded);
    float2 contrast = min(Ldiff/lbkg, 1000.f);

    if (scale != 0) contrast *= 2;

    //apply gaussian precompute (notably CSF)
    const float ch_gain = (channel == 1) ? 1.45f : 1.f;

    const float S = csfhandle.computeSensitivityGPU(lbkg.x, scale, channel);
    contrast *= S * ch_gain;

    //if (glx == 6 && gly == 4) printf("contrast %f on channel %d and scale %d\n", contrast, channel, scale);
    if (glx < width && gly < height){
        outputPlane1[gly*width+glx] = contrast.x;
        outputPlane2[gly*width+glx] = contrast.y;
    }
}

void fromPyrReduceToCSF(float* mem_d, float* mem_d2, const int64_t pyramidSize, const int64_t width, const int64_t height, const int scale, const int channel, CSF_Handler& csfhandle, hipStream_t stream){
    int th_x = 16;
    int th_y = 16;
    int64_t bl_x = (width+th_x-1)/th_x;
    int64_t bl_y = (height+th_y-1)/th_y;
    fromPyrReduceToCSF_Kernel<<<dim3(bl_x, bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, mem_d2, pyramidSize, width, height, scale, channel, csfhandle);
    GPU_CHECK(hipGetLastError());
}

//separable, but not worth it for a kernel of size 5
__global__ void gaussPyrReduce_Kernel(float* dst, float* src, int64_t source_width, int64_t source_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    if (thid >= nw*nh) return;

    //in new space
    const int x = thid%nw;
    const int y = thid/nw;

    float nval = 0.f;

    for (int dx = -2; dx <= 2; dx++){
        const float kernel_x = gaussPyrKernel[dx+2];
        int ref_ind_x = 2*x + dx;
        //symmetric padding
        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= source_width) ref_ind_x = 2*source_width - ref_ind_x - 2;
        for (int dy = -2; dy <= 2; dy++){
            const float kernel_y = gaussPyrKernel[dy+2];
            int ref_ind_y = 2*y + dy;
            //symmetric padding
            if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
            if (ref_ind_y >= source_height) ref_ind_y = 2*source_height - ref_ind_y - 2;

            nval += kernel_x*kernel_y*src[ref_ind_y*source_width+ref_ind_x];
            //if (y*nw+x == 0) printf("Gpyr from %f at %dx%d\n", src[ref_ind_y*source_width+ref_ind_x], ref_ind_x, ref_ind_y);
        }
    }

    dst[y*nw+x] = nval;
    //if (y*nw+x == 0) printf("Gpyr got %f at width %lld\n", nval, source_width);
}

void gaussPyrReduce(float* dst, float* src, int64_t source_width, int64_t source_height, hipStream_t stream){
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    
    int th_x = 256;
    int64_t bl_x = (nw*nh+th_x-1)/th_x;
    gaussPyrReduce_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, source_width, source_height);
    GPU_CHECK(hipGetLastError());
}

//separable, but not worth it for a kernel of size 5
template<bool subdst, bool adddst = false, bool clampMin = false>
__global__ void gaussPyrExpand_Kernel(float* dst, float* src, int64_t new_width, int64_t new_height){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    int ow = (new_width+1)/2;
    int oh = (new_height+1)/2;
    if (thid >= new_width*new_height) return;

    //in new space
    const int x = thid%new_width;
    const int y = thid/new_width;

    float nval = 0.f;

    int parity_x = x%2;
    int parity_y = y%2;

    for (int dx = -2+parity_x; dx <= 2; dx+=2){
        const float kernel_x = 2*gaussPyrKernel[dx+2];
        int ref_ind_x = (x + dx)/2; //funny: x+dx is always even
        if (ref_ind_x < 0) ref_ind_x = -ref_ind_x-1;
        if (ref_ind_x >= ow) ref_ind_x = 2*ow - ref_ind_x -2;
        for (int dy = -2+parity_y; dy <= 2; dy+=2){
            const float kernel_y = 2*gaussPyrKernel[dy+2];
            int ref_ind_y = (y+dy)/2; //(y+dy) is always even
            if (ref_ind_y < 0) ref_ind_y = -ref_ind_y-1;
            if (ref_ind_y >= oh) ref_ind_y = 2*oh - ref_ind_y -2;

            nval += kernel_x*kernel_y*src[ref_ind_y*ow+ref_ind_x];
        }
    }

    if constexpr (clampMin){
        nval = max(nval, 0.01f);
    }

    //if (thid == 0) printf("GlayerEx: %f at %lld\n", nval, new_width);

    if constexpr (subdst) {
        dst[thid] -= nval;
        //if (thid == 0) printf("Layer got %f at width %lld from %f - %f\n", dst[0], new_width, dst[0]+nval, nval);
    } else if constexpr(adddst) {
        dst[thid] += nval;
    } else {
        dst[thid] = nval;
        //if (thid == 13*1024 + 64 && new_width*new_height == 1184264) printf("We got %f at width %lld from %f\n", dst[thid] , new_width, src[y/2*ow+x/2]);
        //if (thid == 0) printf("Layer got %f at width %lld from %f\n", dst[0] , new_width, src[0]);
    }
}

template<bool subdst, bool adddst = false, bool clampMin = false>
void gaussPyrExpand(float* dst, float* src, int64_t new_width, int64_t new_height, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    gaussPyrExpand_Kernel<subdst, adddst, clampMin><<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, src, new_width, new_height);
    GPU_CHECK(hipGetLastError());
}

template<bool isMean, int multiplier>
__global__ void baseBandPyrRefine_Kernel(float* dst, float* p, float* Lbkg, int64_t width){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= width) return;

    //if (thid == 0) printf("Layer is %f at width %lld\n", p[thid], width);

    float val;
    if constexpr (!isMean){
        val = min(p[thid]/max(0.01f, Lbkg[thid]), 1000.f);
        //if (thid == 13*1024 + 64 && width == 1920*1080) printf("baseBandPyrRefine %f at width %d\n", p[thid], width);
    } else {
        //if (thid == 0) printf("value %f %f\n", Lbkg[0], p[thid]);
        //then our adress is the mean, a single float at 0
        val = min(p[thid]/max(0.01f, Lbkg[0]), 1000.f);
    }
    dst[thid] = val*multiplier;
    //if (thid == 0) printf("baseBandPyrRefine %f at width %lld\n", p[thid], width);
}

//gets the contrast from the layers
template<bool isMean, int multiplier>
void baseBandPyrRefine(float* dst, float* p, float* Lbkg, int64_t width, hipStream_t stream){
    int th_x = 256;
    int64_t bl_x = (width + th_x-1)/th_x;
    baseBandPyrRefine_Kernel<isMean, multiplier><<<dim3(bl_x), dim3(th_x), 0, stream>>>(dst, p, Lbkg, width);
    GPU_CHECK(hipGetLastError());
}

//in float
int64_t LpyrMemoryNeedPerPlane(int64_t width, int64_t height, float ppd){
    std::vector<float> bands = get_frequencies(width, height, ppd);
    int64_t res = 0;
    int64_t w = width;
    int64_t h = height;
    for (uint level = 0; level < bands.size(); level++){
        res += h*w;

        w = (w+1)/2;
        h = (h+1)/2;
    }
    return res;
}

}
