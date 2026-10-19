namespace butter{

//Butteraugli's opsin dynamics XYB of src (linear RGB) into dst, which must not be src: opsinDynamicsBlur blurs the 3
//planes (window 3) and combines them with the image in one pass.
void placeOpsinDynamicsImage(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src[3], VkDeviceAddress dst[3], uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, float intensity_multiplier){
    VkPipeline pipeline = deviceHelper->cachedPipeline["separableBlur3Butter.spv:opsinDynamicsBlur"];
    uint8_t kernelArguments[128];
    //0.0f is the 0 that each blur sum starts from
    fillKernelArguments(0, kernelArguments, src[0], src[1], src[2], dst[0], dst[1], dst[2], width, height, gaussianHandle.get(0), gaussianHandle.getIntegral(0), 0.0f, intensity_multiplier);
    //32x64 outputs per workgroup, as separableBlur
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (width-1)/32+1, (height-1)/64+1, 1);
    std::vector<MemoryBarrierPipeline> barriers(6);
    for (uint i = 0; i < 3; i++){
        barriers[2*i] = {src[i], height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT};
        barriers[2*i+1] = {dst[i], height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT};
    }
    deviceHelper->device.placeMemoryBarriers(cmdBuff, barriers);
}

}
