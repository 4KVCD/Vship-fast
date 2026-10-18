#pragma once

namespace ssimu2{

//Halves the linear RGB planes src into dst and converts src to positive XYB; dst too when lastScale.
void placeDownsample(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress src2, VkDeviceAddress src3, VkDeviceAddress dst1, VkDeviceAddress dst2, VkDeviceAddress dst3, uint64_t width, uint64_t height, bool lastScale){
    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    int64_t th_x = 16;
    int64_t th_y = 16;
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;

    const uint32_t lastScaleInt = lastScale ? 1u : 0u;
    const uint32_t zero = 0;

    VkPipeline pipeline = deviceHelper->cachedPipeline["downsampleSSIMU2.spv:downsample"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, src2, src3, dst1, dst2, dst3, width, height, lastScaleInt, zero);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src1, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src2, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src3, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst1, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst2, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dst3, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}
