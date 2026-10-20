#pragma once

#include <vector>
#include <mutex>
#include <string>
#include <filesystem>

#include "../../util/VshipExceptions.hpp"
#define THROW_ERROR(STRING) throw VshipError(HIPError, __FILE__, __LINE__, STRING);
#include "vulkanHelper.hpp"
#undef THROW_ERROR

#include "../../util/VshipExceptions.hpp"
#include <unordered_map>

//special file created that contains every kernels instead of needing files next to vship!
#include "libvshipSpvShaders.hpp"

#define EXPORTVSHIPLIB
#include "../../VshipAPI.h"

//if set to false means that the gpu does not support it

typedef struct InternalDeviceInfo{
    char name[256];
    int integrated; //iGPU? (boolean)
    bool suitable; //1 true 0 false
    Vship_VulkanFeatureMatrix vulkanFeatureMatrix; //only set for Vulkan
} InternalDeviceInfo;

std::filesystem::path shaderDirectory = "libvshipSpvShaders";

//Preloaded at device initialization to avoid repeated harddrive access and compilation
const std::unordered_map<std::string, std::vector<std::string>> entry_pointsPerShader = {
    {"Planed.spv", {"StrideEliminator", "StrideAdder"}},
    {"kernelTest.spv", {"main"}},
    {"basicOperations.spv", {"multArray", "subArray", "memSet"}},
    {"anyDepthToFloat.spv", {"convertToFloatPlane", "repackRows"}},
    {"chromaUpsample.spv", {
        "bicubicHorizontalCenterUpscaleX2", 
        "bicubicHorizontalLeftUpscaleX2", 
        "bicubicHorizontalCenterUpscaleX4", 
        "bicubicHorizontalLeftUpscaleX4",
        "bicubicVerticalCenterUpscaleX2",
        "bicubicVerticalTopUpscaleX2",
        "bicubicVerticalCenterUpscaleX4",
        "bicubicVerticalTopUpscaleX4"
        }
    },
    {"resize.spv", {"horizontalResizeTranspose"}},
    {"primaries.spv", {"primariesToPrimaries"}},
    {"YUVToLinRGB.spv", {"YUVToLinRGB", "YUVToLinRGBUpsampled"}},
    //SSIMU2
    {"downsampleSSIMU2.spv", {"downsample", "yuvToPyramid8", "yuvToPyramid16"}},
    {"makeXYBSSIMU2.spv", {"rgb_to_positive_xyb"}},
    {"scoreSSIMU2.spv", {"blurRows", "blurColumnsAndScore"}},
    //Butter
    {"separableBlur2Butter.spv", {"separableBlur2", "opsinDynamicsBlur"}},
    {"separableBlur3Butter.spv", {"separableBlur3"}},
    {"separableBlur6Butter.spv", {"separableBlur6"}},
    {"separableBlur7Butter.spv", {"separableBlur7"}},
    {"separableBlur16Butter.spv", {"separableBlur16"}},
    {"downupsampleButter.spv", {"addsupersample2X"}},
    {"maltaDiffButter.spv", {"MaltaDiffMapLF", "MaltaDiffMap"}},
    {"maltaDiffFp64Butter.spv", {"MaltaDiffMapLF", "MaltaDiffMap"}},
    {"simplerDiffButter.spv", {"samenoisediff", "diffclamp", "L2diff", "L2diffs"}},
    {"maskPsychoButter.spv", {"fuzzyerrosion", "maskinit"}},
    {"combineMasksButter.spv", {"computeDiffmap"}},
    {"combineMasksFp64Butter.spv", {"computeDiffmap"}},
    {"diffnormsButter.spv", {"sumreducenorm", "sumreduce"}},
    {"separateFrequenciesButter.spv", {
        "supressXbyY"
        }
    },
    //CVVDP
    {"distmap_specificsCVVDP.spv", {"JODIZE", "mergeChromaNorm"}},
    {"lpyrCVVDP.spv", {"baseBandPyrRefine", "gaussPyrExpand", "gaussPyrReduce"}},
    {"fromLpyrToCSFCVVDP.spv", {"fromPyrReduceToCSF"}},
    {"poolingCVVDP.spv", {"reduceSum", "reduceSums"}},
    {"colorsCVVDP.spv", {"displayToDKL", "yuvToDKL"}},
    {"temporalFilterCVVDP.spv", {"temporalConvolutionKernel_d"}},
    {"maskingModelCVVDP.spv", {"computeD_baseband", "computeD", "preGaussianPreCompute"}},
    {"resizeCVVDP.spv", {"horizontalResizeTranspose"}},
};

struct VshipDeviceHelper{
    //should be readonly, address is 'shaderName':'entrypoint'
    std::unordered_map<std::string, VkPipeline> cachedPipeline;
    //can be usedas expected
    VkDeviceManager device;
    //the GPU's PCI vendor ID (VkPhysicalDeviceProperties::vendorID), for kernels whose best workgroup size differs
    uint32_t vendorID;
    VshipDeviceHelper(VkInstance instance, VkPhysicalDevice& hd) : device(instance, hd) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(hd, &properties);
        vendorID = properties.vendorID;
        //auto init = std::chrono::high_resolution_clock::now();
        for (const auto& [shaderName, entry_points] : entry_pointsPerShader) {
            //the double versions of shaders (see fp64Shader) only on GPUs with 64-bit floats
            if (shaderName.find("Fp64") != std::string::npos && !device.hasFloat64) continue;
            std::vector<char>& embedded = shaderNameToByteCode[shaderName];
            std::vector<char> forMetal;
            if (device.isMoltenVK) forMetal = metalShaders::rewrite(embedded, metalShaders::keepsFastMath(shaderName));
            VkShaderModule shaderModule = device.loadShaderModule(device.isMoltenVK ? forMetal : embedded);
            //std::cout << shaderName << " " << shaderNameToByteCode[shaderName].size() << std::endl;
            for (const std::string& entry_point: entry_points){
                cachedPipeline[shaderName + ":" + entry_point] = device.createComputePipeline(shaderModule, entry_point);
            }
            device.freeShaderModule(shaderModule);
        }
        //auto fin = std::chrono::high_resolution_clock::now();
        //uint64_t millitaken = std::chrono::duration_cast<std::chrono::milliseconds>(fin - init).count();
        //std::cout << "Pipeline Creation took " << millitaken << " ms" << std::endl;
    }
    //The pipeline "<name>Fp64Butter.spv:<entry>" on GPUs with 64-bit floats, else "<name>Butter.spv:<entry>": the
    //double version computes like libjxl, the float one approximates it.
    VkPipeline fp64Shader(const std::string& name, const std::string& entry){
        return cachedPipeline[name + (device.hasFloat64 ? "Fp64Butter.spv:" : "Butter.spv:") + entry];
    }
    ~VshipDeviceHelper(){
#ifndef _WIN32
        for (const auto& [path, pipeline]: cachedPipeline){
            device.destroyComputePipeline(pipeline);
        }
#endif
    }
};

//lazy loaded devices
class VkDeviceManagerList{
    VkManager vkInstance;
    std::vector<VkPhysicalDevice> hwlist;
    std::vector<VshipDeviceHelper*> devices;
    std::mutex devicesLock;
    std::vector<uint> cached_suitable; //0 undefined 1 invalid 2 valid
    std::mutex cached_suitableLock;
    bool getSuitable(uint i){
        //when writing this we need to lock
        cached_suitableLock.lock();
        try{
            if (cached_suitable[i] == 0) cached_suitable[i] = (isDeviceSuitable(hwlist[i])) ? 2 : 1;
        } catch (std::runtime_error& e){
            cached_suitableLock.unlock();
            throw VshipError(HIPError, __FILE__, __LINE__, e.what());
        }
        cached_suitableLock.unlock();
        return (cached_suitable[i] == 2) ? true : false;
    }
public:
    VkDeviceManagerList(){
        //vkInstance is loaded by default initialization
        //we use raw device list to expose these to the user
        hwlist = vkInstance.listGPU(true);
        devices.resize(hwlist.size(), NULL);
        cached_suitable.resize(hwlist.size());
    }
    void getDeviceInfo(InternalDeviceInfo* deviceInfo, uint i){
        if (i >= hwlist.size()) throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(hwlist[i], &deviceProperties);

        uint copysize = std::min(256u, VK_MAX_PHYSICAL_DEVICE_NAME_SIZE);
        memcpy(deviceInfo->name, deviceProperties.deviceName, copysize);
        deviceInfo->name[255] = '\0';
        deviceInfo->integrated = deviceProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        deviceInfo->suitable = getSuitable(i);

        //fill vulkan feature matrix
        VkPhysicalDeviceVulkan11Features extension1;
        extension1.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        extension1.pNext = NULL;
        VkPhysicalDeviceVulkan12Features extension2;
        extension2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        extension2.pNext = &extension1;
        VkPhysicalDeviceSynchronization2Features extension3;
        extension3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
        extension3.pNext = &extension2;
        extension3.synchronization2 = true;
        VkPhysicalDeviceFeatures2 deviceFeatures;
        deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures.pNext = &extension3;
        vkGetPhysicalDeviceFeatures2(hwlist[i], &deviceFeatures);

        deviceInfo->vulkanFeatureMatrix.shaderStorageBufferArrayDynamicIndexing = deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing;
        deviceInfo->vulkanFeatureMatrix.shaderFloat64 = deviceFeatures.features.shaderFloat64;
        deviceInfo->vulkanFeatureMatrix.shaderInt64 = deviceFeatures.features.shaderInt64;
        deviceInfo->vulkanFeatureMatrix.shaderInt16 = deviceFeatures.features.shaderInt16;
        deviceInfo->vulkanFeatureMatrix.logicOp = deviceFeatures.features.logicOp;
        deviceInfo->vulkanFeatureMatrix.synchronization2 = extension3.synchronization2;
        deviceInfo->vulkanFeatureMatrix.bufferDeviceAddress = extension2.bufferDeviceAddress;
        deviceInfo->vulkanFeatureMatrix.shaderInt8 = extension2.shaderInt8;
        deviceInfo->vulkanFeatureMatrix.shaderFloat16 = extension2.shaderFloat16;
        deviceInfo->vulkanFeatureMatrix.uniformAndStorageBuffer8BitAccess = extension2.uniformAndStorageBuffer8BitAccess;
        deviceInfo->vulkanFeatureMatrix.storageBuffer8BitAccess = extension2.storageBuffer8BitAccess;
        deviceInfo->vulkanFeatureMatrix.storagePushConstant8 = extension2.storagePushConstant8;
        deviceInfo->vulkanFeatureMatrix.variablePointers = extension1.variablePointers;
        deviceInfo->vulkanFeatureMatrix.variablePointersStorageBuffer = extension1.variablePointersStorageBuffer;
        deviceInfo->vulkanFeatureMatrix.uniformAndStorageBuffer16BitAccess = extension1.uniformAndStorageBuffer16BitAccess;
        deviceInfo->vulkanFeatureMatrix.storageBuffer16BitAccess = extension1.storageBuffer16BitAccess;
    }
    uint getGpuCount(){
        return hwlist.size();
    }
    VshipDeviceHelper* get(uint i){
        if (i >= hwlist.size()) throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
        if (!getSuitable(i)) throw VshipError(BadDeviceCode, __FILE__, __LINE__);
        devicesLock.lock();
        if (devices[i] == NULL) {
            try{
                devices[i] = new VshipDeviceHelper(vkInstance.instance, hwlist[i]);
            } catch(std::runtime_error& e){
                devicesLock.unlock();
                throw VshipError(HIPError, __FILE__, __LINE__, e.what());
            }
        }
        devicesLock.unlock();
        return devices[i];
    }
    ~VkDeviceManagerList(){
        for (int i = 0; i < devices.size(); i++){
            if (devices[i] != NULL) {
                delete devices[i];
            }
        }
    }
};
