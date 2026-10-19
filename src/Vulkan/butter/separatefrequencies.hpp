namespace butter{

void placeSubarray_removerangearound0(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress first, VkDeviceAddress second, uint64_t width, float w){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:subarray_removerangearound0"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, first, second, width, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {first, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {second, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeRemoverangearound0(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress first, uint64_t width, float w){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;
    
    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:removerangearound0"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, first, width, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{first, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}

void placeSubarray_amplifyrangearound0(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress first, VkDeviceAddress second, uint64_t width, float w){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:subarray_amplifyrangearound0"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, first, second, width, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {first, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {second, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeSupressXbyY(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress first, VkDeviceAddress second, uint64_t width, float yw){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:supressXbyY"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, first, second, width, yw);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {first, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {second, sizeof(float)*width, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeSeparateHf_Uhf(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress hf, VkDeviceAddress uhf, uint64_t width){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:separateHf_Uhf"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, hf, uhf, width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {hf, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {uhf, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeXybLowFreqToVals(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress xplane, VkDeviceAddress yplane, VkDeviceAddress bplane, uint64_t width){
    uint64_t th_x = 256;
    uint64_t bl_x = (width-1)/th_x + 1;

    VkPipeline pipeline = deviceHelper->cachedPipeline["separateFrequenciesButter.spv:XybLowFreqToVals"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, xplane, yplane, bplane, width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {xplane, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {yplane, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {bplane, sizeof(float)*width, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeSeparateFrequencies(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src[3], VkDeviceAddress temp[3], VkDeviceAddress lf[3], VkDeviceAddress mf[3], VkDeviceAddress hf[2], VkDeviceAddress uhf[2], uint64_t width, uint64_t height, GaussianHandle& gaussianHandle){
    for (int i = 0; i < 3; i++){
        //we separate lf to get mf BUT we put mf on hf if i != 2 for later reasons
        //the blur also writes xyb-lf, to temp[0] for i == 2, else to uhf[i]
        placeBlur(deviceHelper, cmdBuff, lf[i], src[i], temp[i], width, height, gaussianHandle, 4, (i == 2) ? temp[0] : uhf[i]);

        if (i == 2){
            //mf = blur(xyb-lf)
            placeBlur(deviceHelper, cmdBuff, mf[i], temp[0], temp[1], width, height, gaussianHandle, 3);
            break;
        }
        //mf (hf (uhf)) = xyb-lf //mf is stored on hf which is stored in uhf
        //same thing here, we will put the real hf into uhf to avoid later copy (the blur above wrote it)
        //mf = blur(mf (hf (uhf))) //we blur mf BUT mf is on hf which is on uhf. After this, mf is stored in mf but hf is on uhf
        //the real mf is blurred and we avoid the need to copy the unblurred mf to hf (uhf)
        placeBlur(deviceHelper, cmdBuff, mf[i], uhf[i], temp[i], width, height, gaussianHandle, 3);

        //hf (uhf) = op(mf, hf (uhf))
        if (i == 0){
            placeSubarray_removerangearound0(deviceHelper, cmdBuff, mf[i], uhf[i], width*height, 0.29f); 
        } else {
            placeSubarray_amplifyrangearound0(deviceHelper, cmdBuff, mf[i], uhf[i], width*height, 0.1f);
        }

    }
    //using uhf which contains hf in reality
    placeSupressXbyY(deviceHelper, cmdBuff, uhf[0], uhf[1], width*height, 46.0f);

    for (int i = 0; i < 2; i++){
        //original does uhf = hf but hf is already in uhf.
        //next is hf = blur(hf (uhf)) -> hf is now at its place and uhf has the old hf copy
        placeBlur(deviceHelper, cmdBuff, hf[i], uhf[i], temp[i], width, height, gaussianHandle, 1);

        if (i == 0){
            placeSubarray_removerangearound0(deviceHelper, cmdBuff, hf[i], uhf[i], width*height, 1.5f);
            placeRemoverangearound0(deviceHelper, cmdBuff, uhf[i], width*height, 0.04f);
        } else {
            placeSeparateHf_Uhf(deviceHelper, cmdBuff, hf[i], uhf[i], width*height);
        }
    }

    placeXybLowFreqToVals(deviceHelper, cmdBuff, lf[0], lf[1], lf[2], width*height);        
}

}
