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

//dst += w*(src1 - src2)^2 as libjxl's Mask adds its kMaskToErrorMul term (L2diff in simplerDiff.slang)
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

//L2diffs (simplerDiff.slang): placeGetdiffmap's L2 differences in one pass (L2DiffAsymmetric of the high frequencies,
//L2Diff of the medium ones, SetL2Diff of the low ones after their XybLowFreqToVals). planes is placeGetdiffmap's plane
//memory, ac and dc its 3 contiguous block_diff_ac and block_diff_dc planes; dc need not be set to 0 first. asymWeights
//are the L2DiffAsymmetric weights (w_0gt1, w_0lt1) of the channels X and Y.
void placeL2diffs(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress planes, VkDeviceAddress ac, VkDeviceAddress dc, uint64_t width, const float asymWeights[4], const float mfWeights[3], const float lfWeights[3]){
    const uint64_t th_x = 256;
    const uint64_t bl_x = (width-1)/th_x + 1;
    //libjxl's L2DiffAsymmetric scales its weights by 0.8, in double
    float a[4];
    for (int i = 0; i < 4; i++) a[i] = asymWeights[i]*0.8;

    VkPipeline pipeline = deviceHelper->cachedPipeline["simplerDiffButter.spv:L2diffs"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, planes, ac, dc, width, a[0], a[1], a[2], a[3], mfWeights[0], mfWeights[1], mfWeights[2], lfWeights[0], lfWeights[1], lfWeights[2]);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {ac, 3*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {dc, 3*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {planes, 21*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}