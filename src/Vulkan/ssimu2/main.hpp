#pragma once

#include "../util/preprocessor.hpp"
#include "../../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/basicOperations.hpp"
#include "../../util/concurrency.hpp"
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

void placeDebugTransfer(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, uint8_t* out, VkDeviceAddress in, size_t size){
    deviceHelper->device.memCpyDtoH(cmdBuff, out, in, size);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{in, size, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}});
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
//output of size 108 float
void placeSsimu2GPUProcess(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, float* output, VkDeviceAddress src1_d[3], VkDeviceAddress src2_d[3], VkDeviceAddress temp_d, int64_t width, int64_t height, GaussianHandle& gaussianhandle){
    const int64_t totalscalesize = getTotalScaleSize(width, height);

    //step 1 : fill the downsample part
    int64_t nw = width;
    int64_t nh = height;
    int64_t index = 0;
    for (int scale = 1; scale <= 5; scale++){
        placeDownsample(deviceHelper, cmdBuff, src1_d[0]+sizeof(float)*index, src1_d[1]+sizeof(float)*index, src1_d[2]+sizeof(float)*index, src1_d[0]+sizeof(float)*(index+nw*nh), src1_d[1]+sizeof(float)*(index+nw*nh), src1_d[2]+sizeof(float)*(index+nw*nh), nw, nh, getScaleDownsampleInfo(scale));
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    index = 0;
    for (int scale = 1; scale <= 5; scale++){
        placeDownsample(deviceHelper, cmdBuff, src2_d[0]+sizeof(float)*index, src2_d[1]+sizeof(float)*index, src2_d[2]+sizeof(float)*index, src2_d[0]+sizeof(float)*(index+nw*nh), src2_d[1]+sizeof(float)*(index+nw*nh) , src2_d[2]+sizeof(float)*(index+nw*nh), nw, nh, getScaleDownsampleInfo(scale));
        index += nw*nh;
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    //placeRgb_to_positive_xyb(deviceHelper, cmdBuff, src1_d, totalscalesize);
    //placeRgb_to_positive_xyb(deviceHelper, cmdBuff, src2_d, totalscalesize);

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    //std::vector<float> allscore_res;
    //measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i]
    placeAllscore_map(deviceHelper, cmdBuff, output, src1_d, src2_d, temp_d, width, height, gaussianhandle);

    //step 6 : format the vector

    //step 7 : enjoy !
    //const double ssim = final_score(allscore_res);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    //return ssim;
}

class SSIMU2ComputingImplementation{
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    uint64_t width;
    uint64_t height;

    VshipDeviceHelper* deviceHelper;
    VkCommandPoolManager* commandPool; //contains 3 buffers -> preprocess, runtime frame To Gpu, runtime ssimu2 process
    VkDeviceAddress mem_d;
    VkDeviceAddress converterTempMemory;
    FenceManager* fence;
    VkSemaphore semaphore;
    float* outputCPU;
public:
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id){
        this->deviceHelper = getGlobalVulkanInstance().get(gpu_id);
        commandPool = deviceHelper->device.createCmdPool(3, 0);
        auto& cmdBuffs = commandPool->primaryCommandBuffers;
        this->width = VshipColorConvert::getColorspaceWidth(source_colorspace);
        this->height = VshipColorConvert::getColorspaceHeight(source_colorspace);
        const int64_t totalscalesize = getTotalScaleSize(width, height);
        const uint64_t tempsize = tempAllocsizeScore(width, height);

        //memory allocations
        mem_d = deviceHelper->device.mallocDevice(sizeof(float)*totalscalesize*6 + sizeof(float)*tempsize);

        VkDeviceAddress src1_d[3] = {mem_d, mem_d+sizeof(float)*totalscalesize, mem_d+2*sizeof(float)*totalscalesize};
        VkDeviceAddress src2_d[3] = {mem_d+3*sizeof(float)*totalscalesize, mem_d+4*sizeof(float)*totalscalesize, mem_d+5*sizeof(float)*totalscalesize};
        VkDeviceAddress temp_d = mem_d+6*sizeof(float)*totalscalesize; //of size tempsize
        outputCPU = (float*)deviceHelper->device.mallocHost(108*sizeof(float));

        fence = deviceHelper->device.createFence();
        semaphore = deviceHelper->device.createSemaphores();

        //here we define the final destination of our color converted frames
        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, deviceHelper, src1_d);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, deviceHelper, src2_d);

        size_t tempSizeConverter1 = VshipColorConvert::getTempMemoryNeeded(source_colorspace);
        size_t tempSizeConverter2 = VshipColorConvert::getTempMemoryNeeded(source_colorspace2);
        converterTempMemory = deviceHelper->device.mallocDevice(tempSizeConverter1+tempSizeConverter2);

        //assert they have the same width/height
        if (converter2.getWidth() != width || converter2.getHeight() != height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        //preprocess
        beginCmdBuffer(cmdBuffs[0]);
        gaussianhandle.init(deviceHelper, cmdBuffs[0]);
        endCmdBuffer(cmdBuffs[0]);
        //execute
        deviceHelper->device.executeBuffer(cmdBuffs[0], fence);

        //while it's running, (even though it should take less time on gpu than queuesubmit oof) we do other stuff
        //record buffer2 (the heart of ssimu2)
        beginCmdBuffer(cmdBuffs[2]);
        //color convert
        converter1.placeConvert(cmdBuffs[2], converterTempMemory);
        converter2.placeConvert(cmdBuffs[2], converterTempMemory+tempSizeConverter1);
        //result should be in src1_d src2_d
        placeSsimu2GPUProcess(deviceHelper, cmdBuffs[2], outputCPU, src1_d, src2_d, temp_d, width, height, gaussianhandle);
        endCmdBuffer(cmdBuffs[2]);

        fence->synchronize();
    }
    void destroy(){
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        delete commandPool;
        delete fence;
        deviceHelper->device.destroySemaphores(semaphore);
        deviceHelper->device.freeDevice(mem_d);
        deviceHelper->device.freeDevice(converterTempMemory);
        deviceHelper->device.freeHost((void*)outputCPU);
    }
    double run(const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        auto& cmdBuffs = commandPool->primaryCommandBuffers;

        //create bufferToGPU
        resetCmdBuffer(cmdBuffs[1]);
        beginCmdBuffer(cmdBuffs[1]);
        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        converter1.placeToGPUBuffer(cmdBuffs[1], srcp1, lineSize, converterTempMemory);
        converter2.placeToGPUBuffer(cmdBuffs[1], srcp2, lineSize2, converterTempMemory+tempSizeConverter1);
        endCmdBuffer(cmdBuffs[1]);

        //now we can execute cmdBuffs1 and 2 but synchronized by semaphore
        deviceHelper->device.executeBuffers(cmdBuffs.data()+1, &semaphore, 2, fence);
        //then we have nothing else to do that wait
        fence->synchronize();
        //output should be in outputCPU
        double res = final_score(outputCPU);
        
        return res;
    }
};

}
