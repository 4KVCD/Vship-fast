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

void placeDebugTransfer(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, uint8_t* out, VkDeviceAddress in, size_t size){
    deviceHelper->device.memCpyDtoH(cmdBuff, out, in, size);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{in, size, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}});
}

//The 5 smaller scales of a linear RGB image, and all 6 in positive XYB: planes hold the full scale and room for the
//others after it (getTotalScaleSize floats each).
void placeDownsampleScales(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress planes[3], int64_t width, int64_t height){
    int64_t nw = width;
    int64_t nh = height;
    int64_t index = 0;
    for (int scale = 1; scale <= 5; scale++){
        placeDownsample(deviceHelper, cmdBuff, planes[0]+sizeof(float)*index, planes[1]+sizeof(float)*index, planes[2]+sizeof(float)*index, planes[0]+sizeof(float)*(index+nw*nh), planes[1]+sizeof(float)*(index+nw*nh), planes[2]+sizeof(float)*(index+nw*nh), nw, nh, scale == 5);
        index += nw*nh;
        nw = (nw - 1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
}

//The image's 6 scales in positive XYB into planes: in one pass from the converter's luma and chroma when the
//converter allows it, else the converter's own passes then the downsampling ones.
void placeImageScales(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VshipColorConvert::Converter& converter, VkDeviceAddress converterTemp, VkDeviceAddress planes[3], int64_t width, int64_t height){
    VshipColorConvert::Converter::PixelConversion conversion;
    if (converter.outputIsPixelConversion() && converter.placeChromaForPixelConversion(cmdBuff, converterTemp, conversion)){
        placeYuvToPyramid(deviceHelper, cmdBuff, conversion, planes, width, height);
        return;
    }
    converter.placeConvert(cmdBuff, converterTemp);
    placeDownsampleScales(deviceHelper, cmdBuff, planes, width, height);
}

class SSIMU2ComputingImplementation{
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    uint64_t width;
    uint64_t height;

    VshipDeviceHelper* deviceHelper;
    VkCommandPoolManager* commandPool; //contains 3 buffers -> preprocess, runtime frame To Gpu, runtime ssimu2 process
    VkCommandPoolManager* uploadPool; //one buffer for the copies of the input frames
    VkSemaphore uploadSemaphore;
    VkDeviceAddress mem_d;
    VkDeviceAddress converterTempMemory;
    FenceManager* fence;
    VkSemaphore semaphore;
    float* sumsCPU;
public:
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id){
        this->deviceHelper = getGlobalVulkanInstance().get(gpu_id);
        commandPool = deviceHelper->device.createCmdPool(3, 0);
        uploadPool = deviceHelper->device.createUploadCmdPool(1);
        auto& cmdBuffs = commandPool->primaryCommandBuffers;
        this->width = VshipColorConvert::getColorspaceWidth(source_colorspace);
        this->height = VshipColorConvert::getColorspaceHeight(source_colorspace);
        const int64_t totalscalesize = getTotalScaleSize(width, height);
        const uint64_t tempsize = tempAllocsizeScore(width, height);
        const uint64_t sumssize = sumsSize(width, height);

        //memory allocations
        mem_d = deviceHelper->device.mallocDevice(sizeof(float)*totalscalesize*6 + sizeof(float)*tempsize + sizeof(float)*sumssize);

        VkDeviceAddress src1_d[3] = {mem_d, mem_d+sizeof(float)*totalscalesize, mem_d+2*sizeof(float)*totalscalesize};
        VkDeviceAddress src2_d[3] = {mem_d+3*sizeof(float)*totalscalesize, mem_d+4*sizeof(float)*totalscalesize, mem_d+5*sizeof(float)*totalscalesize};
        VkDeviceAddress temp_d = mem_d+6*sizeof(float)*totalscalesize; //of size tempsize
        VkDeviceAddress sums_d = temp_d+sizeof(float)*tempsize; //of size sumssize
        sumsCPU = (float*)deviceHelper->device.mallocHost(sumssize*sizeof(float));

        fence = deviceHelper->device.createFence();
        semaphore = deviceHelper->device.createSemaphores();
        uploadSemaphore = deviceHelper->device.createSemaphores();

        //here we define the final destination of our color converted frames
        //libjxl's input arithmetic: libjxl is the reference of this metric
        converter1.init(source_colorspace, VshipColorConvert::linRGBBT709, deviceHelper, src1_d, true);
        converter2.init(source_colorspace2, VshipColorConvert::linRGBBT709, deviceHelper, src2_d, true);

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
        //color conversion and scales, then the scores
        placeImageScales(deviceHelper, cmdBuffs[2], converter1, converterTempMemory, src1_d, width, height);
        placeImageScales(deviceHelper, cmdBuffs[2], converter2, converterTempMemory+tempSizeConverter1, src2_d, width, height);
        placeAllscore_map(deviceHelper, cmdBuffs[2], sumsCPU, sums_d, src1_d, src2_d, temp_d, width, height, gaussianhandle);
        endCmdBuffer(cmdBuffs[2]);

        fence->synchronize();
    }
    void destroy(){
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        delete commandPool;
        delete uploadPool;
        delete fence;
        deviceHelper->device.destroySemaphores(semaphore);
        deviceHelper->device.destroySemaphores(uploadSemaphore);
        deviceHelper->device.freeDevice(mem_d);
        deviceHelper->device.freeDevice(converterTempMemory);
        deviceHelper->device.freeHost((void*)sumsCPU);
    }
    double run(const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        auto& cmdBuffs = commandPool->primaryCommandBuffers;

        //copies to the GPU on the upload queue, conversion to float at the start of the compute work
        VkCommandBuffer uploadCmd = uploadPool->primaryCommandBuffers[0];
        resetCmdBuffer(uploadCmd);
        beginCmdBuffer(uploadCmd);
        size_t tempSizeConverter1 = converter1.getTempMemoryNeeded();
        converter1.placeUpload(uploadCmd, srcp1, lineSize, converterTempMemory);
        converter2.placeUpload(uploadCmd, srcp2, lineSize2, converterTempMemory+tempSizeConverter1);
        endCmdBuffer(uploadCmd);
        resetCmdBuffer(cmdBuffs[1]);
        beginCmdBuffer(cmdBuffs[1]);
        converter1.placeToFloat(cmdBuffs[1]);
        converter2.placeToFloat(cmdBuffs[1]);
        endCmdBuffer(cmdBuffs[1]);

        //now we can execute cmdBuffs1 and 2 but synchronized by semaphore
        deviceHelper->device.executeUploadThenBuffers(uploadCmd, uploadSemaphore, cmdBuffs.data()+1, &semaphore, 2, fence);
        //then we have nothing else to do that wait
        fence->synchronize();
        //the sums should be in sumsCPU
        double res = final_score(sumsCPU, width, height);
        
        return res;
    }
};

}
