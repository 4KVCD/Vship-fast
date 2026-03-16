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
void placeGetDistMap(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress res[4], std::vector<int64_t> bandOffsets, int64_t width, int64_t height, int levels){
    //first step is merging chroma planes
    int64_t w = width;
    int64_t h = height;
    for (int band = 0; band < levels; band++){
        if (band == 0 || band == levels-1){
            placeMergeChromaNorm(deviceHelper, cmdBuff, res[0]+bandOffsets[band], res[1]+bandOffsets[band], res[2]+bandOffsets[band], res[3]+bandOffsets[band], w*h, 1);
        } else {
            placeMergeChromaNorm(deviceHelper, cmdBuff, res[0]+bandOffsets[band], res[1]+bandOffsets[band], res[2]+bandOffsets[band], res[3]+bandOffsets[band], w*h, 2);
        }
        w = (w+1)/2;
        h = (h+1)/2;
    }
    //reconstruction
    w = width;
    h = height;
    for (int band = levels-2; band >= 0; band--){
        placeGaussPyrExpand<false, true, false>(deviceHelper, cmdBuff, res[0]+bandOffsets[band], res[0]+bandOffsets[band+1], w, h);
        w = (w+1)/2;
        h = (h+1)/2;
    }

    VkPipeline pipeline = deviceHelper->cachedPipeline["distmap_specificsCVVDP.spv:JODIZE"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, res[0], (int64_t)(width*height));
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (width*height+255)/256, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {{res[0], width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
}

}