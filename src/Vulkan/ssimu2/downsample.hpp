namespace ssimu2{

void placeDownsample(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress dst, uint64_t width, uint64_t height){
    int64_t newh = (height-1)/2 + 1;
    int64_t neww = (width-1)/2 + 1;

    int64_t th_x = 16;
    int64_t th_y = 16;
    int64_t bl_x = (neww-1)/th_x + 1;
    int64_t bl_y = (newh-1)/th_y + 1;
    VkPipeline pipeline = deviceHelper->cachedPipeline["downsampleSSIMU2.spv:downsample"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src, dst, width, height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}