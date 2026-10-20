namespace butter{

//libjxl's MaltaDiffMapT weights (lib/jxl/butteraugli/butteraugli.cc): w_pre in double, then norm2 = w_pre*norm1
//rounded to float, as the kernel uses them
inline void maltaWeights(double w_0gt1, double w_0lt1, double norm1, double mulli, float& norm2_0gt1, float& norm2_0lt1){
    const double len = 3.75;
    const float kWeight0 = 0.5f;
    const float kWeight1 = 0.33f;
    const double w_pre0gt1 = mulli * std::sqrt(kWeight0 * w_0gt1) / (len * 2 + 1);
    const double w_pre0lt1 = mulli * std::sqrt(kWeight1 * w_0lt1) / (len * 2 + 1);
    norm2_0gt1 = (float)(w_pre0gt1 * norm1);
    norm2_0lt1 = (float)(w_pre0lt1 * norm1);
}

//block_diff_ac += libjxl's MaltaDiffMap (entry "MaltaDiffMap", mulli 0.39905817637) or MaltaDiffMapLF (entry
//"MaltaDiffMapLF", mulli 0.611612573796) of lum0 and lum1, with libjxl's double weights
void placeMalta(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, const char* entry, double mulli, VkDeviceAddress lum0, VkDeviceAddress lum1, VkDeviceAddress block_diff_ac, const uint64_t width, const uint64_t height, double w_0gt1, double w_0lt1, double norm1){
    float norm2_0gt1, norm2_0lt1;
    maltaWeights(w_0gt1, w_0lt1, norm1, mulli, norm2_0gt1, norm2_0lt1);

    //32x32 outputs per workgroup (MALTA_TILE in maltaDiff.slang)
    const uint64_t bl_x = (width - 1)/32 + 1;
    const uint64_t bl_y = (height - 1)/32 + 1;

    VkPipeline pipeline = deviceHelper->fp64Shader("maltaDiff", entry);
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, lum0, lum1, block_diff_ac, width, height, norm2_0gt1, norm2_0lt1, (float)norm1, 0u); //0u: the run-time zero of the double-float steps
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {lum0, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {lum1, height*width*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {block_diff_ac, height*width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeMaltaDiffMap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress lum0, VkDeviceAddress lum1, VkDeviceAddress block_diff_ac, const uint64_t width, const uint64_t height, double w_0gt1, double w_0lt1, double norm1){
    placeMalta(deviceHelper, cmdBuff, "MaltaDiffMap", 0.39905817637, lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1);
}

void placeMaltaDiffMapLF(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress lum0, VkDeviceAddress lum1, VkDeviceAddress block_diff_ac, const uint64_t width, const uint64_t height, double w_0gt1, double w_0lt1, double norm1){
    placeMalta(deviceHelper, cmdBuff, "MaltaDiffMapLF", 0.611612573796, lum0, lum1, block_diff_ac, width, height, w_0gt1, w_0lt1, norm1);
}

}
