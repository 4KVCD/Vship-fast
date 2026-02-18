namespace butter{

void placeMaltaDiffMap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress lum0, VkDeviceAddress lum1, VkDeviceAddress block_diff_ac, const uint64_t width, const uint64_t height, const float w_0gt1, const float w_0lt1, const float norm1){
    const float len = 3.75f;
    const float mulli = 0.39905817637f;

    const uint64_t th_x = 16;
    const uint64_t th_y = 16;
    const uint64_t bl_x = (width - 1)/th_x + 1;
    const uint64_t bl_y = (height - 1)/th_y + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maltaDiffButter.spv:MaltaDiffMap"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {lum0, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {lum1, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeMaltaDiffMapLF(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress lum0, VkDeviceAddress lum1, VkDeviceAddress block_diff_ac, const uint64_t width, const uint64_t height, const float w_0gt1, const float w_0lt1, float norm1){
    const float len = 3.75f;
    const float mulli = 0.611612573796f;

    const uint64_t th_x = 16;
    const uint64_t th_y = 16;
    const uint64_t bl_x = (width - 1)/th_x + 1;
    const uint64_t bl_y = (height - 1)/th_y + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maltaDiffButter.spv:MaltaDiffMapLF"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1, len, mulli);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {lum0, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {lum1, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}