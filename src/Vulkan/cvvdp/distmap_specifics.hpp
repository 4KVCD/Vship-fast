#pragma once

namespace cvvdp{

float toJOD(float a){
    if (a > 0.1f){
        return 10.f - jod_a * powf(a, jod_exp);
    } else {
        const float jod_a_p = jod_a * (powf(0.1f, jod_exp-1.f));
        return 10.f - jod_a_p * a;
    }
}

float topow4(float el){
    float res = el;
    res *= res;
    res *= res;
    return res;
}

void placeMergeChromaNorm(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress p0, VkDeviceAddress p1, VkDeviceAddress p2, VkDeviceAddress p3, int64_t size, float divisor){
    int64_t th_x = 256;
    int64_t bl_x = (size+th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["distmap_specificsCVVDP.spv:mergeChromaNorm"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, p0, p1, p2, p3, size, divisor);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {p0, size*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {p1, size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {p2, size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {p3, size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//we should have D values in Lpyr bands and channels.
//Lbkg planes will serve as a buffer and the final result willl be in Lbkg band 0
void placeGetDistMap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, LpyrManager& Lpyr){
    //first step is merging chroma planes
    for (int band = 0; band < Lpyr.getSize(); band++){
        const auto [w, h] = Lpyr.getResolution(band);
        if (band == 0 || band == Lpyr.getSize()-1){
            placeMergeChromaNorm(deviceHelper, cmdBuff, Lpyr.getContrast(0, band), Lpyr.getContrast(1, band), Lpyr.getContrast(2, band), Lpyr.getContrast(3, band), w*h, 1);
        } else {
            placeMergeChromaNorm(deviceHelper, cmdBuff, Lpyr.getContrast(0, band), Lpyr.getContrast(1, band), Lpyr.getContrast(2, band), Lpyr.getContrast(3, band), w*h, 2);
        }
    }
    //reconstruction
    for (int band = Lpyr.getSize()-2; band >= 0; band--){
        const auto [w, h] = Lpyr.getResolution(band);
        placeGaussPyrExpand<false, true, false>(deviceHelper, cmdBuff, Lpyr.getContrast(0, band), Lpyr.getContrast(0, band+1), w, h);
    }
    const auto [w0, h0] = Lpyr.getResolution(0);
    deviceHelper->device.memCpyDtoD(cmdBuff, Lpyr.getLbkg(0), Lpyr.getContrast(0, 0), sizeof(float)*w0*h0);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {Lpyr.getLbkg(0), w0*h0*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {Lpyr.getContrast(0, 0), w0*h0*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });

    VkPipeline pipeline = deviceHelper->cachedPipeline["distmap_specificsCVVDP.spv:JODIZE"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, Lpyr.getLbkg(0), (int64_t)(w0*h0));
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (w0*h0+255)/256, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{Lpyr.getLbkg(0), w0*h0*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}

}