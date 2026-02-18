#pragma once

namespace VshipColorConvert{

void placePrimariesToPrimaries(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress p0, VkDeviceAddress p1, VkDeviceAddress p2, int64_t width, Vship_Primaries_t src_primary, Vship_Primaries_t dst_primary){
    if (dst_primary == src_primary) return;

    const int thx = 256;
    const int blx = (width + thx -1)/thx;
    
    VkPipeline pipeline = deviceHelper->cachedPipeline["primaries.spv:primariesToPrimaries"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, (uint64_t)p0, (uint64_t)p1, (uint64_t)p2, (uint64_t)width, (int)src_primary, (int)dst_primary);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p0, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p1, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p2, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}