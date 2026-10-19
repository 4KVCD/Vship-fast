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

//The converter's conversion to linear RGB (as prepared by Converter::placeChromaForPixelConversion) and
//placeDisplayToDKL in one pass (yuvToDKL in shaders/colors.slang), into the DKL planes out of a width*height image
void placeYUVToDKL(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, const VshipColorConvert::Converter::PixelConversion& conversion, VkDeviceAddress out[3], int64_t width, int64_t height, float Y_peak, float Y_black, float Y_refl, float exposure, Vship_TransferFunction_t transferFunction, DisplayColorspace isHDR, Vship_Primaries_t primaries){
    const int th_x = 256;
    const int64_t bl_x = (width+th_x-1)/th_x;
    //an SDR display shows the input as sRGB whatever its actual transfer function
    const int pq = (isHDR && transferFunction == Vship_TRC_PQ) ? 1 : 0;
    const uint32_t zero = 0;

    VkPipeline pipeline = deviceHelper->cachedPipeline["colorsCVVDP.spv:yuvToDKL"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, conversion.luma, conversion.chroma1, conversion.chroma2, out[0], out[1], out[2], (uint64_t)width, (uint64_t)height, conversion.verticalMode, (int)conversion.matrix, (int)conversion.transfer, (int)conversion.libjxlArithmetic, Y_peak, Y_black, Y_refl, exposure, pq, (int)primaries, zero);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, height, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {out[0], width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out[1], width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out[2], width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {conversion.luma, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {conversion.chroma1, width*conversion.chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {conversion.chroma2, width*conversion.chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}
