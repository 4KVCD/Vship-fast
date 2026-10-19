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

}