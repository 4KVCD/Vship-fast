#pragma once

namespace VshipColorConvert{

void placeYUVToLinRGB(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress p0, VkDeviceAddress p1, VkDeviceAddress p2, int64_t width, Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer, bool libjxlArithmetic = false){
    if (transfer == Vship_TRC_Linear && matrix == Vship_MATRIX_RGB) return;
    const int thx = 256;
    const int blx = (width + thx -1)/thx;

    VkPipeline pipeline = deviceHelper->cachedPipeline["YUVToLinRGB.spv:YUVToLinRGB"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, p0, p1, p2, width, (int)matrix, (int)transfer, (int)libjxlArithmetic);

    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p0, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p1, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p2, width*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    });
}

//placeYUVToLinRGB on width*height planes, with the chroma's vertical upsampling (verticalMode from
//chromaVerticalMode) done for each pixel: the chroma comes from chroma1 and chroma2 at full width, with chromaHeight
//rows. They may be p1 and p2 when verticalMode is 0.
void placeYUVToLinRGBUpsampled(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress p0, VkDeviceAddress p1, VkDeviceAddress p2, VkDeviceAddress chroma1, VkDeviceAddress chroma2, int64_t width, int64_t height, int64_t chromaHeight, int verticalMode, Vship_YUVMatrix_t matrix, Vship_TransferFunction_t transfer, bool libjxlArithmetic, VkDeviceAddress rawLuma, int rawLumaFormat, float rawLumaScale, size_t rawLumaBytes){
    const int thx = 256;
    const int blx = (width + thx -1)/thx;
    const uint32_t zero = 0;

    VkPipeline pipeline = deviceHelper->cachedPipeline["YUVToLinRGB.spv:YUVToLinRGBUpsampled"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, p0, p1, p2, chroma1, chroma2, (uint64_t)width, (uint64_t)height, verticalMode, (int)matrix, (int)transfer, (int)libjxlArithmetic, zero, rawLuma, rawLumaFormat, rawLumaScale);

    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, blx, height, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p0, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p1, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p2, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {chroma1, width*chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {chroma2, width*chromaHeight*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
    //the luma read from the upload (rawLuma, which the next frame's copies write again)
    if (rawLuma != 0) deviceHelper->device.placeMemoryBarriers(cmdBuff, {{rawLuma, rawLumaBytes, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}});
}

}