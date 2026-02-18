namespace butter{

void placeComputeDiffmap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress mask, VkDeviceAddress block_diff_dc0, VkDeviceAddress block_diff_dc1, VkDeviceAddress block_diff_dc2, VkDeviceAddress block_diff_ac0, VkDeviceAddress block_diff_ac1, VkDeviceAddress block_diff_ac2, VkDeviceAddress dst, uint64_t width){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["combineMasksButter.spv:computeDiffmap"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, mask, block_diff_dc0, block_diff_dc1, block_diff_dc2, block_diff_ac0, block_diff_ac1, block_diff_ac2, dst, width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {block_diff_dc0, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_dc1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_dc2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac0, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

}