#pragma once

namespace cvvdp{

__device__ __host__ float toJOD(float a){
    if (a > 0.1f){
        return 10.f - jod_a * powf(a, jod_exp);
    } else {
        const float jod_a_p = jod_a * (powf(0.1f, jod_exp-1.f));
        return 10.f - jod_a_p * a;
    }
}

__device__ float inline topow4(float el){
    float res = el;
    res *= res;
    res *= res;
    return res;
}

template<int divisor>
__global__ void mergeChromaNorm_Kernel(float* p0, float* p1, float* p2, float* p3, int64_t size){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= size) return;

    p0[thid] = powf((topow4(p0[thid]) + topow4(p1[thid]) + topow4(p2[thid]) + topow4(p3[thid]))*0.25, 0.25f)/(float)divisor;
}

__global__ void JODIZE_Kernel(float* p, int64_t size){
    const int64_t thid = threadIdx.x + blockIdx.x * blockDim.x;
    if (thid >= size) return;

    p[thid] = 1.f - (toJOD(p[thid]))*0.1f;
}

template<int divisor>
void mergeChromaNorm(float* p0, float* p1, float* p2, float* p3, int64_t size, hipStream_t stream){
    int th_x = 256;
    int bl_x = (size+th_x-1)/th_x;
    mergeChromaNorm_Kernel<divisor><<<dim3(bl_x), dim3(th_x), 0, stream>>>(p0, p1, p2, p3, size);
    GPU_CHECK(hipGetLastError());
}

//output in res[0]
void getDistMap(float* res[4], std::vector<int64_t>& bandOffsets, int64_t width, int64_t height, int levels, hipStream_t stream){
    //first step is merging chroma planes
    int64_t w = width;
    int64_t h = height;
    for (int band = 0; band < levels; band++){
        if (band == 0 || band == levels-1){
            mergeChromaNorm<1>(res[0]+bandOffsets[band], res[1]+bandOffsets[band], res[2]+bandOffsets[band], res[2]+bandOffsets[band], w*h, stream);
        } else {
            mergeChromaNorm<2>(res[0]+bandOffsets[band], res[1]+bandOffsets[band], res[2]+bandOffsets[band], res[2]+bandOffsets[band], w*h, stream);
        }
        w = (w+1)/2;
        h = (h+1)/2;
    }
    //reconstruction
    w = width;
    h = height;
    for (int band = levels-2; band >= 0; band--){
        gaussPyrExpand<false, true, false>(res[0]+bandOffsets[band], res[0]+bandOffsets[band+1], w, h, stream);
        w = (w+1)/2;
        h = (h+1)/2;
    }
    JODIZE_Kernel<<<dim3((width*height+255)/256), dim3(256), 0, stream>>>(res[0], width*height);
    GPU_CHECK(hipGetLastError());
}

}