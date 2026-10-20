namespace butter{

void placeSamenoisediff(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress dst, uint64_t width, float w){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:samenoisediff"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, dst, width, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeDiffclamp(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress src2, VkDeviceAddress dst, uint64_t width, float maxclamp){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:diffclamp"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, src2, dst, width, maxclamp);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeL2diff(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress src2, VkDeviceAddress dst, uint64_t width, float w){
    if (w == 0) return;
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:L2diff"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, src2, dst, width, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeL2AsymDiff(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src1, VkDeviceAddress src2, VkDeviceAddress dst, uint64_t width, float w_0gt1, float w_0lt1){
    if (w_0gt1 == 0.0f && w_0lt1 == 0.0f) return;
    w_0gt1 *= 0.8;
    w_0lt1 *= 0.8;
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:L2AsymDiff"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src1, src2, dst, width, w_0gt1, w_0lt1);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src1, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {src2, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//L2diffs (simplerDiff.slang): placeGetdiffmap's L2diff updates in one pass, after placeL2AsymDiff, with the low
//frequencies' XybLowFreqToVals. planes is placeGetdiffmap's plane memory, ac and dc its 3 contiguous block_diff_ac and
//block_diff_dc planes; dc need not be set to 0 first.
void placeL2diffs(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress planes, VkDeviceAddress ac, VkDeviceAddress dc, uint64_t width, const float mfWeights[3], const float lfWeights[3]){
    const uint64_t th_x = 256;
    const uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:L2diffs"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, planes, ac, dc, width, mfWeights[0], mfWeights[1], mfWeights[2], lfWeights[0], lfWeights[1], lfWeights[2], 0u);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {ac, 3*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dc, 3*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {planes, 21*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}