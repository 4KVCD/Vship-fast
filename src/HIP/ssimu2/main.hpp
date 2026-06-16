#pragma once

#include "../util/preprocessor.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"
#include "makeXYB.hpp"
#include "downsample.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

namespace ssimu2{

int64_t getTotalScaleSize(int64_t width, int64_t height){
    int64_t result = 0;
    for (int scale = 0; scale < 6; scale++){
        result += width*height;
        width = (width-1)/2+1;
        height = (height-1)/2+1;
    }
    return result;
}

//scale goes from 1 to 5 included (size of the small scale)
downsampleKernelInformation getScaleDownsampleInfo(int scale){
    downsampleKernelInformation info;
    info.result_XYB = (scale == 5); //last scale only

    SkipMap skipMaps[3];
    for (int i = 0; i < 3; i++){
        skipMaps[i] = getSkipMap(i, scale-1);
    }
    info.skipX = skipMaps[0].ssim && skipMaps[0].artifact && skipMaps[0].detailloss;
    info.skipY = skipMaps[1].ssim && skipMaps[1].artifact && skipMaps[1].detailloss;
    info.skipB = skipMaps[2].ssim && skipMaps[2].artifact && skipMaps[2].detailloss;

    return info;
}

//expects planar linear RGB input. Beware that each src1_d, src2_d and temp_d must be of size "totalscalesize" even if the actual image is contained in a width*height format
double ssimu2GPUProcess(float* src1_d[3], float* src2_d[3], float* temp_d, int64_t width, int64_t height, GaussianHandle& gaussianhandle, hipStream_t streams[2], hipEvent_t events[4]){
    const int64_t totalscalesize = getTotalScaleSize(width, height);

    //step 1 : fill the downsample part
    int64_t nw = width;
    int64_t nh = height;
    int64_t index = 0;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d[0]+index, src1_d[1]+index, src1_d[2]+index, src1_d[0]+index+nw*nh, src1_d[1]+index+nw*nh, src1_d[2]+index+nw*nh, nw, nh, getScaleDownsampleInfo(scale), streams[0]);
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    index = 0;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src2_d[0]+index, src2_d[1]+index, src2_d[2]+index, src2_d[0]+index+nw*nh, src2_d[1]+index+nw*nh, src2_d[2]+index+nw*nh, nw, nh, getScaleDownsampleInfo(scale), streams[1]);
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    std::vector<float> allscore_res;
    //measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i]
    try{
        allscore_res = allscore_map(src1_d, src2_d, temp_d, width, height, gaussianhandle, streams, events);
    } catch (const VshipError& e){
        throw e;
    }

    //step 7 : enjoy !
    const double ssim = final_score(allscore_res);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    return ssim;
}

class SSIMU2ComputingImplementation{
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    int64_t width;
    int64_t height;
    hipStream_t streams[2];
    hipEvent_t events[4];
    int gpu_id;
public:
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id){
        this->gpu_id = gpu_id;
        GPU_CHECK(hipSetDevice(gpu_id));
        GPU_CHECK(hipStreamCreate(streams+0));
        GPU_CHECK(hipStreamCreate(streams+1));
        for (int i = 0; i < 4; i++){
            GPU_CHECK(hipEventCreate(events+i));
        }

        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, streams[0]);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, streams[1]);

        this->width = converter1.getWidth();
        this->height = converter1.getHeight();

        //assert they have the same width/height
        if (converter2.getWidth() != width || converter2.getHeight() != height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        gaussianhandle.init();
    }
    void destroy(){
        GPU_CHECK(hipSetDevice(gpu_id));
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        GPU_CHECK(hipStreamDestroy(streams[0]));
        GPU_CHECK(hipStreamDestroy(streams[1]));
        for (int i = 0; i < 4; i++){
            GPU_CHECK(hipEventDestroy(events[i]));
        }
    }
    double run(const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        GPU_CHECK(hipSetDevice(gpu_id));
        const int64_t totalscalesize = getTotalScaleSize(width, height);

        const uint64_t tempsize = tempAllocsizeScore(width, height); //possessed by stream1 by default
        float* mem_d1;
        float* mem_d2;
        hipError_t erralloc;
        erralloc = hipMallocAsync(&mem_d1, sizeof(float)*totalscalesize*3 + sizeof(float)*tempsize, streams[0]); //2 base image and 1 reduction+copy buffer
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        erralloc = hipMallocAsync(&mem_d2, sizeof(float)*totalscalesize*3, streams[1]); //2 base image and 1 reduction+copy buffer
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }

        float* src1_d[3] = {mem_d1, mem_d1+totalscalesize, mem_d1+2*totalscalesize};
        float* src2_d[3] = {mem_d2, mem_d2+totalscalesize, mem_d2+2*totalscalesize};
        float* temp_d = mem_d1+3*totalscalesize; //of size tempsize

        converter1.convert(src1_d, srcp1, lineSize);
        converter2.convert(src2_d, srcp2, lineSize2);
        
        double res;
        try {
            res = ssimu2GPUProcess(src1_d, src2_d, temp_d, width, height, gaussianhandle, streams, events);
        } catch (const VshipError& e){
            (void)hipFree(mem_d1);
            (void)hipFree(mem_d2);
            throw e;
        }

        GPU_CHECK(hipFreeAsync(mem_d1, streams[0]));
        GPU_CHECK(hipFreeAsync(mem_d2, streams[1]));

        return res;
    }
};

}
