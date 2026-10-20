#pragma once

namespace cvvdp{

#include "../util/vulkanDeviceManager.hpp"

//channels firstChannel .. firstChannel+channels-1 of a band in one dispatch; channel 0 must be done before the others
//The contrast of the 4 channels of a band (planes 5 to 8) and the next band's input (written after this band's), for
//both images: fromPyrReduceToCSF.
void placeFromPyrReduceToCSF(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress mem_d, VkDeviceAddress mem_d2, const int64_t pyramidSize, const int64_t width, const int64_t height, const int scale, CSF_Handler& csfhandle){
    //the dispatch's planes of a kind are pyramidSize apart
    const int64_t planesSpan = pyramidSize*3;
    const int64_t small_width = (width+1)/2;
    const int64_t small_height = (height+1)/2;

    int th_x = 16;
    int th_y = 16;
    int64_t bl_x = (width+th_x-1)/th_x;
    int64_t bl_y = (height+th_y-1)/th_y;

    VkPipeline pipeline = deviceHelper->cachedPipeline["fromLpyrToCSFCVVDP.spv:fromPyrReduceToCSF"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, mem_d, mem_d2, csfhandle.mem_d, pyramidSize, width, height, scale);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        //read input planes
        {mem_d, sizeof(float)*(planesSpan+width*height), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {mem_d2, sizeof(float)*(planesSpan+width*height), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},

        //write contrast
        {mem_d+sizeof(float)*pyramidSize*5, sizeof(float)*(planesSpan+width*height), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {mem_d2+sizeof(float)*pyramidSize*5, sizeof(float)*(planesSpan+width*height), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},

        //write smallScale
        {mem_d+sizeof(float)*width*height, sizeof(float)*(planesSpan+small_width*small_height), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {mem_d2+sizeof(float)*width*height, sizeof(float)*(planesSpan+small_width*small_height), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
    });
}

void placeGaussPyrReduce(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t source_width, int64_t source_height){
    int nw = (source_width+1)/2;
    int nh = (source_height+1)/2;
    
    int th_x = 256;
    int64_t bl_x = (nw*nh+th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:gaussPyrReduce"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, source_width, source_height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, nh*nw*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, source_width*source_height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

template<bool subdst, bool adddst = false, bool clampMin = false>
void placeGaussPyrExpand(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t new_width, int64_t new_height){
    int th_x = 256;
    int64_t bl_x = (new_width*new_height+th_x-1)/th_x;
    int64_t old_width = (new_width-1)/2+1;
    int64_t old_height = (new_height-1)/2+1;

    constexpr uint conditions = (clampMin ? 4 : 0)|(adddst ? 2 : 0)|(subdst ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:gaussPyrExpand"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, new_width, new_height, conditions);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, new_width*new_height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, old_width*old_height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//gets the contrast from the layers
template<bool isMean, int multiplier>
void placeBaseBandPyrRefine(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress p, VkDeviceAddress Lbkg, int64_t width){
    int th_x = 256;
    int64_t bl_x = (width + th_x-1)/th_x;

    constexpr uint conditions = (isMean ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["lpyrCVVDP.spv:baseBandPyrRefine"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, p, Lbkg, width, conditions, multiplier);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {Lbkg, (isMean ? 1 : width)*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
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
