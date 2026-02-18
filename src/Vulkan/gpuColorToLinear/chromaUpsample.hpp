#pragma once

namespace VshipColorConvert{

//temp is of size 2 planes
//source is of size width * height possibly chroma downsampled
int placeChromaUpsample(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress temp, VkDeviceAddress src[3], uint64_t width, uint64_t height, Vship_ChromaLocation_t location, Vship_ChromaSubsample_t subsampledata){
    const int subw = subsampledata.subw;
    const int subh = subsampledata.subh;
    
    if (subw == 0 && subh == 0) return 0;
    int64_t smallwidth = ((width-1) >> subw)+1;
    int64_t smallheight = ((height-1) >> subh)+1;
    const int thx = 16;
    const int thy = 16;
    int blx = (smallwidth+1 + thx-1)/thx; //will change after upsampling of horizontal
    int bly = (smallheight + thy-1)/thy;

    switch (location){
        case (Vship_ChromaLoc_Left):
        case (Vship_ChromaLoc_TopLeft):
            if (subw == 0){
            } else if (subw == 1){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicHorizontalLeftUpscaleX2"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)temp, (uint64_t)(src[1]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)(src[2]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else if (subw == 2){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicHorizontalLeftUpscaleX4"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)temp, (uint64_t)(src[1]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)(src[2]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Top):
            if (subw == 0){
            } else if (subw == 1){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicHorizontalCenterUpscaleX2"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)temp, (uint64_t)(src[1]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)(src[2]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else if (subw == 2){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicHorizontalCenterUpscaleX4"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)temp, (uint64_t)(src[1]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)(src[2]), (uint64_t)width, (uint64_t)smallheight);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subw != 0) return 1; //not implemented
    }

    //we need to copy the data to temp
    if (subw == 0){
        deviceHelper->device.memCpyDtoD(cmdBuff, temp, src[1], sizeof(float)*width*smallheight);
        deviceHelper->device.memCpyDtoD(cmdBuff, temp+sizeof(float)*width*smallheight, src[2], sizeof(float)*width*smallheight);
    }

    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {temp, sizeof(float)*width*smallheight*2, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src[1], sizeof(float)*smallwidth*smallheight, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src[2], sizeof(float)*smallwidth*smallheight, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });

    if (subh == 0){
        deviceHelper->device.memCpyDtoD(cmdBuff, src[1], temp, sizeof(float)*width*smallheight);
        deviceHelper->device.memCpyDtoD(cmdBuff, src[2], temp+sizeof(float)*width*smallheight, sizeof(float)*width*smallheight);
    }

    blx = (width + thx-1)/thx;
    bly = (smallheight+1 + thy-1)/thy;

    switch (location){
        case (Vship_ChromaLoc_Top):
        case (Vship_ChromaLoc_TopLeft):
            if (subh == 0){
            } else if (subh == 1){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicVerticalTopUpscaleX2"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)src[1], (uint64_t)temp, (uint64_t)width, (uint64_t)height);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)(src[2]), (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)width, (uint64_t)height);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else {
                return 1; //not implemented
            }
            break;
        case (Vship_ChromaLoc_Center):
        case (Vship_ChromaLoc_Left):
            if (subh == 0){
            } else if (subh == 1){
                VkPipeline pipeline = deviceHelper->cachedPipeline["chromaUpsample.spv:bicubicVerticalCenterUpscaleX2"];
                uint8_t kernelArguments[128];
                fillKernelArguments(0, kernelArguments, (uint64_t)src[1], (uint64_t)temp, (uint64_t)width, (uint64_t)height);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
                fillKernelArguments(0, kernelArguments, (uint64_t)(src[2]), (uint64_t)temp+sizeof(float)*width*smallheight, (uint64_t)width, (uint64_t)height);
                deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, bly, 1);
            } else {
                return 1; //not implemented
            }
            break;
        default:
            if (subh != 0) return 1; //not implemented
    }

    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {temp, sizeof(float)*width*smallheight*2, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src[1], sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src[2], sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
    return 0;
}

}