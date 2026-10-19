#pragma once

namespace cvvdp{

template<int power, bool applyPower, bool applyInversePower>
void placeReduceSum(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress src, int64_t size, bool divide, float scale, bool safePow, bool clampMin){
    constexpr uint th_x = 1024;
    int64_t bl_x = (size+th_x-1)/th_x;

    uint conditions = (clampMin ? 16 : 0)|(safePow ? 8 : 0)|(applyPower ? 4 : 0)|(applyInversePower ? 2 : 0)|(divide ? 1 : 0);
    VkPipeline pipeline = deviceHelper->cachedPipeline["poolingCVVDP.spv:reduceSum"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, dst, src, size, power, conditions, scale);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {dst, bl_x*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {src, size*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}

//the result will be at temp[0]. We suppose that temp is of the same size as src
//The inputs are multiplied by scale (and clamped at 0.01 if clampMin) before the power; safePow uses pycvvdp's
//safe_pow for both powers.
template<int power>
void placeComputeMean(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress src, VkDeviceAddress temp, int64_t size, bool divide, float scale = 1.f, bool safePow = false, bool clampMin = false){
    VkDeviceAddress final_dst = temp; //to contain temp[0]
    VkDeviceAddress tempbuffer[3] = {temp+sizeof(float)*1, temp+sizeof(float)*(1+(size+1023)/1024), src};
    int oscillator = 2; //corresponds to current source except the first time
    while (size > 1024){
        int destination = (oscillator == 2) ? 0 : (oscillator^1);
        if (oscillator == 2){
            placeReduceSum<power, true, false>(deviceHelper, cmdBuff, tempbuffer[destination], tempbuffer[oscillator], size, divide, scale, safePow, clampMin); 
        } else {
            placeReduceSum<power, false, false>(deviceHelper, cmdBuff, tempbuffer[destination], tempbuffer[oscillator], size, false, scale, safePow, clampMin); 
        }
        oscillator = destination;
        size = (size+1023)/1024;
    }
    if (oscillator == 2){
        placeReduceSum<power, true, true>(deviceHelper, cmdBuff, final_dst, tempbuffer[oscillator], size, divide, scale, safePow, clampMin); 
    } else {
        placeReduceSum<power, false, true>(deviceHelper, cmdBuff, final_dst, tempbuffer[oscillator], size, false, scale, safePow, clampMin); 
    }
}

}