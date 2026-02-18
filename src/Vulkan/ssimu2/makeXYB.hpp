#ifndef MAKEXYBHPP
#define MAKEXYBHPP

namespace ssimu2{

void placeRgb_to_positive_xyb(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src_d[3], int64_t width){
    int64_t th_x = 256;
    int64_t bl_x = (width-1)/th_x + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["makeXYBSSIMU2.spv:rgb_to_positive_xyb"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src_d[0], src_d[1], src_d[2], width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src_d[0], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src_d[1], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src_d[2], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}

#endif