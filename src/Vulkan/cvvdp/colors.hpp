#pragma once

namespace cvvdp{

void placeXYZ_to_dkl(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src_d[3], int64_t width){
    int th_x = 256;
    int64_t bl_x = (width+th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["colorsCVVDP.spv:XYZ_to_dkl"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, src_d[0], src_d[1], src_d[2], width);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {src_d[0], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src_d[1], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src_d[2], width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

void placeDisplayEncode(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src_d, int64_t width, float Y_peak, float Y_black, float Y_refl, float exposure, Vship_TransferFunction_t transferFunction, DisplayColorspace isHDR){
    int th_x = 256;
    int64_t bl_x = (width+th_x-1)/th_x;
    if (!isHDR){
        transferFunction = Vship_TRC_BT709; 
        //we supppose that the input is sRGB no matter the actual input since the screen will display it as is
    }
    uint8_t kernelArguments[128];
    VkPipeline pipeline;
    fillKernelArguments(0, kernelArguments, src_d, width, Y_peak, Y_black, Y_refl, exposure);
    switch (transferFunction){
        case Vship_TRC_PQ:
            pipeline = deviceHelper->cachedPipeline["colorsCVVDP.spv:displayEncodePQ"];
            deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
            break;
        //treated same as BT709/sRGB
        case Vship_TRC_Linear:
        case Vship_TRC_HLG:
        default:
            pipeline = deviceHelper->cachedPipeline["colorsCVVDP.spv:displayEncodeBT709"];
            deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
            break;
    }
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{src_d, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}


}