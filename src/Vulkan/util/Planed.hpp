#pragma once

void placeStrideEliminator(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress strided, uint64_t stride, uint64_t width, uint64_t height){
    uint64_t wh = width*height;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["Planed.spv:StrideEliminator"];
    uint64_t kernelArguments[16];
    kernelArguments[0] = (uint64_t)dst;
    kernelArguments[1] = (uint64_t)strided;
    kernelArguments[2] = (uint64_t)stride;
    kernelArguments[3] = (uint64_t)width;
    kernelArguments[4] = (uint64_t)height;
    deviceHelper->device.placeKernel(cmdBuff, pipeline, (uint8_t*)kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {strided, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeStrideAdder(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress strided, VkDeviceAddress src, uint64_t stride, uint64_t width, uint64_t height){
    uint64_t wh = width*height;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["Planed.spv:StrideAdder"];
    uint64_t kernelArguments[16];
    kernelArguments[0] = (uint64_t)strided;
    kernelArguments[1] = (uint64_t)src;
    kernelArguments[2] = (uint64_t)stride;
    kernelArguments[3] = (uint64_t)width;
    kernelArguments[4] = (uint64_t)height;
    deviceHelper->device.placeKernel(cmdBuff, pipeline, (uint8_t*)kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {strided, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}