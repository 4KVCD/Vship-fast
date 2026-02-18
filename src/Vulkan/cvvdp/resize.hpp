#pragma once

namespace cvvdp{

//dest of resize size and temp biggest of the 2 size, src of source size
void placeResizePlane(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dest, VkDeviceAddress temp, VkDeviceAddress src, int64_t source_width, int64_t source_height, int64_t resize_width, int64_t resize_height){
    int th_x = 256;
    int64_t bl1_x = (resize_width*source_height + th_x-1)/th_x;
    int64_t bl2_x = (resize_width*resize_height + th_x-1)/th_x;

    if (source_width == resize_width && source_height == resize_height){
        deviceHelper->device.memCpyDtoD(cmdBuff, dest, src, sizeof(float)*source_width*source_height);
        return;
    }

    VkPipeline pipeline = deviceHelper->cachedPipeline["resizeCVVDP.spv:horizontalResizeTranspose"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, (int64_t)temp, (int64_t)src, (int64_t)source_width, (int64_t)source_height, (int64_t)resize_width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl1_x, 1, 1);

    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {temp, sizeof(float)*resize_width*source_height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, sizeof(float)*source_width*source_height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });

    fillKernelArguments(0, kernelArguments, (int64_t)dest, (int64_t)temp, (int64_t)source_height, (int64_t)resize_width, (int64_t)resize_height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl2_x, 1, 1);

    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src, sizeof(float)*resize_width*resize_height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {temp, sizeof(float)*resize_width*source_height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}