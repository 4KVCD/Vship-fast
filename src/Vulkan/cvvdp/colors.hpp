#pragma once

namespace cvvdp{

//One image's linear RGB planes in to the DKL planes out (displayToDKL in shaders/colors.slang): the display model,
//the image's primaries to XYZ, then DKL. in and out may be the same planes.
void placeDisplayToDKL(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress in[3], VkDeviceAddress out[3], int64_t size, float Y_peak, float Y_black, float Y_refl, float exposure, Vship_TransferFunction_t transferFunction, DisplayColorspace isHDR, Vship_Primaries_t primaries){
    int th_x = 256;
    int64_t bl_x = (size+th_x-1)/th_x;
    //an SDR display shows the input as sRGB whatever its actual transfer function
    const int pq = (isHDR && transferFunction == Vship_TRC_PQ) ? 1 : 0;

    VkPipeline pipeline = deviceHelper->cachedPipeline["colorsCVVDP.spv:displayToDKL"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, in[0], in[1], in[2], out[0], out[1], out[2], size, Y_peak, Y_black, Y_refl, exposure, pq, (int)primaries);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {out[0], size*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out[1], size*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out[2], size*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {in[0], size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {in[1], size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {in[2], size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}
