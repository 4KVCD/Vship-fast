#pragma once

#include "util/gpuhelper.hpp"
#define EXPORTVSHIPLIB
#include "../VshipAPI.h"

static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore * core, const VSAPI *vsapi) {
    (void)userData;
    (void)core;

    std::stringstream ss;
    int count;
    InternalDeviceInfo devattr;

    //we don't need a full check at that point
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }
    
    if (count <= gpuid || gpuid < 0){
        vsapi->mapSetError(out, VshipError(BadDeviceArgument, __FILE__, __LINE__).getErrorMessage().c_str());
        return;
    }

    if (error != peSuccess){
        //no gpu_id was selected
        for (int i = 0; i < count; i++){
            try{
                globalVulkanInstance.getDeviceInfo(&devattr, i);
            } catch (const VshipError& e){
                vsapi->mapSetError(out, e.getErrorMessage().c_str());
                return;
            }
            ss << "GPU " << i << ": " << devattr.name << std::endl;
        }
    } else {
        int passgpuCheck;
        try{
            globalVulkanInstance.getDeviceInfo(&devattr, gpuid);
            VshipDeviceHelper* deviceHelper = globalVulkanInstance.get(gpuid);
            passgpuCheck = (int)helper::gpuKernelCheck(deviceHelper);
        } catch (const VshipError& e){
            vsapi->mapSetError(out, e.getErrorMessage().c_str());
            return;
        }

        ss << "Name: " << devattr.name << std::endl;
        ss << "MultiProcessorCount: " << 0 << std::endl;
        ss << "MaxSharedMemoryPerBlock: " << 0 << " bytes" << std::endl;
        ss << "WarpSize: " << 0 << std::endl;
        ss << "VRAMCapacity: " << 0 << " GB" << std::endl;
        ss << "MemoryBusWidth: " << 0 << " bits" << std::endl;
        ss << "Integrated: " << devattr.integrated << std::endl;
        ss << "PassKernelCheck : " << passgpuCheck << std::endl;
    }
    vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
}

void Vship_GetDeviceInfoImpl(Vship_DeviceInfo* device_info, int gpu_id){
    int count = helper::checkGpuCount();
    if (gpu_id >= count){
        throw VshipError(BadDeviceArgument, __FILE__, __LINE__, "while using Vship_GetDeviceInfo");
    }
    InternalDeviceInfo devattr;
    globalVulkanInstance.getDeviceInfo(&devattr, gpu_id);
    memcpy(device_info->name, devattr.name, 256); //256 char to copy
    //device_info->VRAMSize = devattr.totalGlobalMem;
    device_info->integrated = devattr.integrated;
    //device_info->MultiProcessorCount = devattr.multiProcessorCount;
    //device_info->WarpSize = devattr.warpSize;
    device_info->vulkanFeatureMatrix = devattr.vulkanFeatureMatrix;
}

void Vship_PinnedMallocImpl(void** ptr, uint64_t size, int gpu_id){
    try{
        *ptr = globalVulkanInstance.get(gpu_id)->device.mallocHost(size);
    } catch(std::runtime_error& e){
        throw VshipError(OutOfRAM, __FILE__, __LINE__, e.what());
    }
}

void Vship_PinnedFreeImpl(void* ptr, int gpu_id){
    try{
        globalVulkanInstance.get(gpu_id)->device.freeHost(ptr);
    } catch(std::runtime_error& e){
        throw VshipError(BadPointer, __FILE__, __LINE__, e.what());
    }
}