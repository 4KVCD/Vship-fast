namespace butter{

void placeSumReduce(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, uint64_t sourceNB){
    const uint64_t th_x = 1024;
    const uint64_t bl_x = (sourceNB-1)/th_x+1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["diffnormsButter.spv:sumreduce"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, sourceNB, bl_x);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, sizeof(float)*bl_x*3, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, sizeof(float)*sourceNB*3, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeSumReduceNorm(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, uint64_t width, uint Qnorm){
    const uint64_t th_x = 1024;
    const uint64_t bl_x = (width-1)/th_x+1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["diffnormsButter.spv:sumreducenorm"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, width, bl_x, Qnorm);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, sizeof(float)*bl_x*3, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, sizeof(float)*width, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeDiffmapscore(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress diffmap, VkDeviceAddress temp, VkDeviceAddress temp2, VkDeviceAddress outbuffer, uint64_t width, uint Qnorm){
    uint64_t firstbufferSize = (width-1)/1024+1;
    VkDeviceAddress tmps[2] = {temp, temp2};
    placeSumReduceNorm(deviceHelper, cmdBuff, (firstbufferSize == 1) ? outbuffer:tmps[0], diffmap, width, Qnorm);

    int oscillate = 0; //current valid buffer
    while (firstbufferSize != 1){
        placeSumReduce(deviceHelper, cmdBuff, (firstbufferSize <= 1024) ? outbuffer:tmps[oscillate^1], tmps[oscillate], firstbufferSize);

        oscillate ^= 1;
        firstbufferSize = (firstbufferSize-1)/1024+1;
    }
}

//finishes the norm to execute on CPU runtime
void posttreatment(float* values, uint Qnorm){
    values[0] = std::pow(values[0], 1./(float)Qnorm);
    values[1] = std::pow(values[1], 1./3.);
}

}