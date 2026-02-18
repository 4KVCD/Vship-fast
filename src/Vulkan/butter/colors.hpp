namespace butter{

void placeOpsinDynamicsImage(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src[3], VkDeviceAddress temp[3], uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, float intensity_multiplier){
    //change src from SRGB to opsin dynamic XYB
    int64_t th_x = 256;
    int64_t bl_x = (width*height-1)/th_x + 1;
    //printf("initial adress: %llu\n", (unsigned long long)src[0].mem_d);
    for (int i = 0; i < 3; i++){
        placeBlurDstNoTemp(deviceHelper, cmdBuff, temp[i], src[i], width, height, gaussianHandle, 0);
    }

    VkPipeline pipeline = deviceHelper->cachedPipeline["colorsButter.spv:opsinDynamicsImage"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src[0], src[1], src[2], temp[0], temp[1], temp[2], width, height, intensity_multiplier);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    std::vector<MemoryBarrierPipeline> barriers(6);
    for (uint i = 0; i < 3; i++){
        barriers[2*i] = {src[i], height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT};
        barriers[2*i+1] = {temp[i], height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT};
    }
    deviceHelper->device.placeMemoryBarriers(cmdBuff, barriers);
}

}