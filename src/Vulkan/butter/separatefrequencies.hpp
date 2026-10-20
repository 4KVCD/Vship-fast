namespace butter{

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

//The blurs need no temporary planes. For X and Y, uhf holds xyb-lf until the medium frequency blur, which reads it
//and writes mf and the high frequencies (rawHf, in temp[1] and temp[2]); the window-4 blur reads rawHf and writes hf
//and uhf. A blur cannot write over its source.
void placeSeparateFrequencies(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src[3], VkDeviceAddress temp[3], VkDeviceAddress lf[3], VkDeviceAddress mf[3], VkDeviceAddress hf[2], VkDeviceAddress uhf[2], uint64_t width, uint64_t height, GaussianHandle& gaussianHandle){
    const VkDeviceAddress rawHf[2] = {temp[1], temp[2]};
    for (int i = 0; i < 3; i++){
        //the blur also writes xyb-lf, to temp[0] for i == 2, else to uhf[i]
        placeBlur(deviceHelper, cmdBuff, lf[i], src[i], temp[i], width, height, gaussianHandle, 4, (i == 2) ? temp[0] : uhf[i], BLUR_POST_RESIDUAL);

        if (i == 2){
            //mf = blur(xyb-lf)
            placeBlur(deviceHelper, cmdBuff, mf[i], temp[0], temp[1], width, height, gaussianHandle, 3);
            break;
        }
        //mf = blur(xyb-lf), rawHf = xyb-lf - mf, then the range operation on mf (the former subarray_removerangearound0
        //and subarray_amplifyrangearound0 passes)
        if (i == 0){
            placeBlur(deviceHelper, cmdBuff, mf[i], uhf[i], temp[0], width, height, gaussianHandle, 3, rawHf[i], BLUR_POST_SUBTRACT_REMOVE_RANGE, 0.29f);
        } else {
            placeBlur(deviceHelper, cmdBuff, mf[i], uhf[i], temp[0], width, height, gaussianHandle, 3, rawHf[i], BLUR_POST_SUBTRACT_AMPLIFY_RANGE, 0.1f);
        }
    }
    placeSupressXbyY(deviceHelper, cmdBuff, rawHf[0], rawHf[1], width*height, 46.0f);

    //hf = blur(rawHf) and uhf from rawHf - hf, with the range operations of the passes that followed the blur
    placeBlur(deviceHelper, cmdBuff, hf[0], rawHf[0], temp[0], width, height, gaussianHandle, 1, uhf[0], BLUR_POST_REMOVE_RANGES);
    placeBlur(deviceHelper, cmdBuff, hf[1], rawHf[1], temp[0], width, height, gaussianHandle, 1, uhf[1], BLUR_POST_SEPARATE_HF_UHF);

    //lf is converted by XybLowFreqToVals in placeL2diffs, its only reader
}

}
