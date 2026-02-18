#pragma once

void placeMultArray(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress src2, VkDeviceAddress dst, uint64_t width){
    uint64_t wh = width;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["basicOperations.spv:multArray"];
    uint64_t kernelArguments[16];
    kernelArguments[0] = (uint64_t)dst;
    kernelArguments[1] = (uint64_t)src;
    kernelArguments[2] = (uint64_t)src2;
    kernelArguments[3] = (uint64_t)width;
    deviceHelper->device.placeKernel(cmdBuff, pipeline, (uint8_t*)kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeSubArray(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress src2, VkDeviceAddress dst, uint64_t width){
    uint64_t wh = width;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["basicOperations.spv:subArray"];
    uint64_t kernelArguments[16];
    kernelArguments[0] = (uint64_t)dst;
    kernelArguments[1] = (uint64_t)src;
    kernelArguments[2] = (uint64_t)src2;
    kernelArguments[3] = (uint64_t)width;
    deviceHelper->device.placeKernel(cmdBuff, pipeline, (uint8_t*)kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeMemset(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, uint64_t width, float val){
    uint64_t wh = width;
    uint64_t th_x = 256;
    uint64_t bl_x = (wh-1)/th_x + 1;
    
    VkPipeline pipeline = deviceHelper->cachedPipeline["basicOperations.spv:memSet"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, width, val);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{dst, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}