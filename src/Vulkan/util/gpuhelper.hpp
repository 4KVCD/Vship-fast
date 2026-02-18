#ifndef GPUHELPERHPP
#define GPUHELPERHPP

#include "preprocessor.hpp"
#include "../../util/VshipExceptions.hpp"

//here is the format of the answer:

//case where gpu_id is not specified:

//GPU 0: {GPU Name}
//...

//case where gpu_id is specified:

//Name: {GPU Name string}
//MultiProcessorCount: {multiprocessor count integer}
//ClockRate: {clockRate float} Ghz
//MaxSharedMemoryPerBlock: {Max Shared Memory Per Block integer} bytes
//WarpSize: {Warp Size integer}
//VRAMCapacity: {GPU VRAM Capacity float} GB
//MemoryBusWidth: {memory bus width integer} bits
//MemoryClockRate: {memory clock rate float} Ghz
//Integrated: {0|1}
//PassKernelCheck : {0|1}

namespace helper{

    int checkGpuCount(){
        int count = globalVulkanInstance.getGpuCount();
        if (count == 0){
            throw VshipError(NoDeviceDetected, __FILE__, __LINE__);
        }
        return count;
    }

    bool gpuKernelCheck(VshipDeviceHelper* deviceHelper){
        VkDeviceManager& vkDevice = deviceHelper->device;
        
        //memory test
        VkDeviceAddress address = vkDevice.mallocDevice(4);
        void* address2 = vkDevice.mallocHost(4);
        ((float*)address2)[0] = 78.f;

        //pipeline
        VkPipeline pipeline = deviceHelper->cachedPipeline["kernelTest.spv:main"];
        //cmdpool
        VkCommandPoolManager* cmdPool = vkDevice.createCmdPool(1, 0);
        VkCommandBuffer cmdBuff = cmdPool->primaryCommandBuffers[0];
        beginCmdBuffer(cmdBuff);
        vkDevice.memCpyHtoD(cmdBuff, address, address2, 4);
        vkDevice.placeMemoryBarriers(cmdBuff, {{address, 4, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
        uint8_t kernelArguments[128];
        //put our device address as our first 8 bytes arguments
        fillKernelArguments(0, kernelArguments, (uint64_t)address);
        vkDevice.placeKernel(cmdBuff, pipeline, kernelArguments, 1, 1, 1);
        vkDevice.placeMemoryBarriers(cmdBuff, {{address, 4, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}});
        vkDevice.memCpyDtoH(cmdBuff, address2, address, 4);
        endCmdBuffer(cmdBuff);

        //launching queue and using fence
        FenceManager* fence = vkDevice.createFence();
        vkDevice.executeBuffer(cmdBuff, fence);
        fence->synchronize();

        //does it work? (no fp imprecisions since 78 and 78+1 is exact in float)
        bool res = ((float*)address2)[0] == 79.f;

        //cleanup
        delete fence;
        delete cmdPool;
        vkDevice.freeDevice(address);
        vkDevice.freeHost(address2);
        
        return res;
    }

    void gpuFullCheck(int gpuid = 0){
        VshipDeviceHelper* deviceHelper = globalVulkanInstance.get(gpuid);
        if (!gpuKernelCheck(deviceHelper)){
            throw VshipError(BadDeviceCode, __FILE__, __LINE__);
        }
    }

    std::string listGPU(){
        std::stringstream ss;
        int count = checkGpuCount();
        
        InternalDeviceInfo deviceInfo;
        for (int i = 0; i < count; i++){
            globalVulkanInstance.getDeviceInfo(&deviceInfo, i);
            ss << "GPU " << i << ": " << deviceInfo.name << std::endl;
        }
        return ss.str();
    }
}

#endif