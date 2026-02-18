#pragma once

#include "util/gpuhelper.hpp"
#define EXPORTVSHIPLIB
#include "../VshipAPI.h"

static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore * core, const VSAPI *vsapi) {
    (void)userData;
    (void)core;

    std::stringstream ss;
    int count, device;
    hipDeviceProp_t devattr;

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
            GPU_CHECK(hipSetDevice(i));
            GPU_CHECK(hipGetDevice(&device));
            GPU_CHECK(hipGetDeviceProperties(&devattr, device));
            ss << "GPU " << i << ": " << devattr.name << std::endl;
        }
    } else {
        GPU_CHECK(hipSetDevice(gpuid));
        GPU_CHECK(hipGetDevice(&device));
        GPU_CHECK(hipGetDeviceProperties(&devattr, device));
        ss << "Name: " << devattr.name << std::endl;
        ss << "MultiProcessorCount: " << devattr.multiProcessorCount << std::endl;
        //ss << "ClockRate: " << ((float)devattr.clockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda 13
        ss << "MaxSharedMemoryPerBlock: " << devattr.sharedMemPerBlock << " bytes" << std::endl;
        ss << "WarpSize: " << devattr.warpSize << std::endl;
        ss << "VRAMCapacity: " << ((float)devattr.totalGlobalMem)/1000000000 << " GB" << std::endl;
        ss << "MemoryBusWidth: " << devattr.memoryBusWidth << " bits" << std::endl;
        //ss << "MemoryClockRate: " << ((float)devattr.memoryClockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda13
        ss << "Integrated: " << devattr.integrated << std::endl;
        ss << "PassKernelCheck : " << (int)helper::gpuKernelCheck() << std::endl;
    }
    vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
}

void Vship_GetDeviceInfoImpl(Vship_DeviceInfo* device_info, int gpu_id){
    int count = helper::checkGpuCount();
    if (gpu_id >= count){
        throw VshipError(BadDeviceArgument, __FILE__, __LINE__, "while using Vship_GetDeviceInfo");
    }
    hipDeviceProp_t devattr;
    GPU_CHECK(hipGetDeviceProperties(&devattr, gpu_id));
    
    memcpy(device_info->name, devattr.name, 256); //256 char to copy
    device_info->VRAMSize = devattr.totalGlobalMem;
    device_info->integrated = devattr.integrated;
    device_info->MultiProcessorCount = devattr.multiProcessorCount;
    device_info->WarpSize = devattr.warpSize;
}

void Vship_PinnedMallocImpl(void** ptr, uint64_t size, int gpu_id){
    GPU_CHECK(hipHostMalloc(ptr, size));
}

void Vship_PinnedFreeImpl(void* ptr, int gpu_id){
    GPU_CHECK(hipHostFree(ptr));
}