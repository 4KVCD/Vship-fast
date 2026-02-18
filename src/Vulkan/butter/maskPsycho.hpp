namespace butter{

void placeDiffPrecompute(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress dst, uint64_t width, uint64_t height, float mul, float bias_arg){
    uint64_t th_x = 256;
    uint64_t bl_x = (width*height-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskPsychoButter.spv:diffPrecompute"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src, dst, width, height, mul, bias_arg);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeMaskinit(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress hfx, VkDeviceAddress uhfx, VkDeviceAddress hfy, VkDeviceAddress uhfy, VkDeviceAddress dst, uint64_t width, float a, float b, float c){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskPsychoButter.spv:maskinit"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, hfx, uhfx, hfy, uhfy, dst, width, a, b, c);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {hfx, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {uhfx, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {hfy, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {uhfy, width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeFuzzyerrosion(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress dst, uint64_t width, uint64_t height){
    uint64_t th_x = 256;
    uint64_t bl_x = (width*height-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskPsychoButter.spv:fuzzyerrosion"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src, dst, width, height);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeMaskPsychoImage(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress hf1[3], VkDeviceAddress uhf1[3], VkDeviceAddress hf2[3], VkDeviceAddress uhf2[3], VkDeviceAddress mask_xyb0, VkDeviceAddress mask_xyb1, VkDeviceAddress mask, VkDeviceAddress block_diff_ac[3], uint64_t width, uint64_t height, GaussianHandle& gaussianHandle) {
    const float muls[3] = {
      2.5f,
      0.4f,
      0.4f,
    };

    placeMaskinit(deviceHelper, cmdBuff, hf1[0], uhf1[0], hf1[1], uhf1[1], mask_xyb0, width*height, muls[0], muls[1], muls[2]);
    placeMaskinit(deviceHelper, cmdBuff, hf2[0], uhf2[0], hf2[1], uhf2[1], mask_xyb1, width*height, muls[0], muls[1], muls[2]);
    
    //hf and uhf are not used anymore and can serve as temporary planes
    VkDeviceAddress diff0 = hf1[0];
    VkDeviceAddress diff1 = hf1[1];

    placeDiffPrecompute(deviceHelper, cmdBuff, mask_xyb0, diff0, width, height, 6.19424080439f, 12.61050594197f);
    placeDiffPrecompute(deviceHelper, cmdBuff, mask_xyb1, diff1, width, height, 6.19424080439f, 12.61050594197f);
    VkDeviceAddress blurred0 = mask_xyb0;
    VkDeviceAddress blurred1 = mask_xyb1;
    placeBlurDstNoTemp(deviceHelper, cmdBuff, blurred0, diff0, width, height, gaussianHandle, 2);
    placeBlurDstNoTemp(deviceHelper, cmdBuff, blurred1, diff1, width, height, gaussianHandle, 2);
    placeFuzzyerrosion(deviceHelper, cmdBuff, blurred0, mask, width, height);
    placeL2diff(deviceHelper, cmdBuff, blurred0, blurred1, block_diff_ac[1], width*height, 10.0f);
}

}