namespace butter{

void inline placeDownsample(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress dst, uint64_t width, uint64_t height){
    uint64_t newh = (height-1)/2 + 1;
    uint64_t neww = (width-1)/2 + 1;

    uint64_t th_x = 16;
    uint64_t th_y = 16;
    uint64_t bl_x = (neww-1)/th_x + 1;
    uint64_t bl_y = (newh-1)/th_y + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["downupsampleButter.spv:downsample"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src, dst, width, height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void inline placeAddsupersample2X(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress diffmap, VkDeviceAddress diffmapsmall, uint64_t width, uint64_t height, float w){
    uint64_t th_x = 16;
    uint64_t th_y = 16;
    uint64_t bl_x = (width-1)/th_x + 1;
    uint64_t bl_y = (height-1)/th_y + 1;

    uint64_t newh = (height-1)/2 + 1;
    uint64_t neww = (width-1)/2 + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["downupsampleButter.spv:addsupersample2X"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, diffmap, diffmapsmall, width, height, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {diffmap, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {diffmapsmall, neww*newh*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}