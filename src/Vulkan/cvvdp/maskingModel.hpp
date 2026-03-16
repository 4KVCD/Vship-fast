#pragma once

namespace cvvdp{

void placePreGaussianPreCompute(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress Lbkg, VkDeviceAddress p1, VkDeviceAddress p2, int64_t width, int64_t height, int channel, int band, CSF_Handler& csfhandle){
    int64_t th_x = 256;
    int64_t bl_x = (width*height +th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskingModelCVVDP.spv:preGaussianPreCompute"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, Lbkg, p1, p2, csfhandle.mem_d, width, height, channel, band);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {Lbkg, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeComputeD(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress out0, VkDeviceAddress out1, VkDeviceAddress out2, VkDeviceAddress out3, VkDeviceAddress R0, VkDeviceAddress R1, VkDeviceAddress R2, VkDeviceAddress R3, VkDeviceAddress T0, VkDeviceAddress T1, VkDeviceAddress T2, VkDeviceAddress T3, const int64_t width, const int64_t height, GaussianHandle& gaussianhandle){
    int th_x = 16;
    int th_y = 16;
    int bl_x = (width +th_x-1)/th_x;
    int bl_y = (height +th_y-1)/th_y;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskingModelCVVDP.spv:computeD"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, out0, out1, out2, out3, R0, R1, R2, R3, T0, T1, T2, T3, width, height, gaussianhandle.gaussiankernel_d, gaussianhandle.gaussiankernel_integral_d);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, bl_y, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {out0, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {out3, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {R0, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {R1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {R2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {R3, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {T0, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {T1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {T2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {T3, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

void placeComputeD_baseband(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress out, VkDeviceAddress Lbkg, VkDeviceAddress p1, VkDeviceAddress p2, int width, int height, int channel, int band, CSF_Handler& csfhandle){
    int th_x = 256;
    int bl_x = (width*height +th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["maskingModelCVVDP.spv:computeD_baseband"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, out, Lbkg, p1, p2, csfhandle.mem_d, width, height, channel, band);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {out, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p1, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {p2, sizeof(float)*width*height, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {Lbkg, sizeof(float)*1, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

}