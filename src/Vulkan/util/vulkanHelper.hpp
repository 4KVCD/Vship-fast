#pragma once

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <string>
#include <fstream>
#include <map>
#include <mutex>

#ifdef FP16SUPPORT
constexpr bool fp16Support = true;
#else
constexpr bool fp16Support = false;
#endif

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include <vulkan/vulkan.h>

using uint = unsigned int;

std::vector<const char*> required_extensions = {
    
};

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    }

    return VK_FALSE;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr; // Optional
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamily{
    std::optional<uint32_t> indices;
    uint maxHWQueueCount;
};

//choose so that we have the maximum number of concurrent queues available
QueueFamily findQueueFamilies(const VkPhysicalDevice& device) {
    QueueFamily out;
    
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    int max_queueCount = 0;
    uint32_t condition = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    for (const auto& queueFamily : queueFamilies) {
        //std::cout << queueFamily.queueCount << std::endl;
        if (queueFamily.queueFlags & condition && queueFamily.queueCount > max_queueCount) {
            out.indices = i;
            out.maxHWQueueCount = queueFamily.queueCount;
            max_queueCount = queueFamily.queueCount;
            //break;
        }

        i++;
    }
    //std::cout << "choose :" << out.indices.value() << " with " << out.maxHWQueueCount << std::endl;
    
    return out;
}

VkPhysicalDeviceFeatures requiredFeaturesBasic = {
    false, //robustBufferAccess
    false, //fullDrawIndexUint32
    false, //imageCubeArray
    false, //independentBlend
    false, //geometryShader
    false, //tessellationShader
    false, //sampleRateShading
    false, //dualSrcBlend
    true, //logicOp
    false, //multiDrawIndirect
    false, //drawIndirectFirstInstance
    false, //depthClamp
    false, //depthBiasClamp
    false, //fillModeNonSolid
    false, //depthBounds
    false, //wideLines
    false, //largePoints
    false, //alphaToOne
    false, //multiViewport
    false, //samplerAnisotropy
    false, //textureCompressionETC2
    false, //textureCompressionASTC_LDR
    false, //textureCompressionBC
    false, //occlusionQueryPrecise
    false, //pipelineStatisticsQuery
    false, //vertexPipelineStoresAndAtomics
    false, //fragmentStoresAndAtomics
    false, //shaderTessellationAndGeometryPointSize
    false, //shaderImageGatherExtended
    false, //shaderStorageImageExtendedFormats
    false, //shaderStorageImageMultisample
    false, //shaderStorageImageReadWithoutFormat
    false, //shaderStorageImageWriteWithoutFormat
    false, //shaderUniformBufferArrayDynamicIndexing
    false, //shaderSampledImageArrayDynamicIndexing
    true, //shaderStorageBufferArrayDynamicIndexing
    false, //shaderStorageImageArrayDynamicIndexing
    false, //shaderClipDistance
    false, //shaderCullDistance
    false, //shaderFloat64
    true, //shaderInt64
    true, //shaderInt16
    false, //shaderResourceResidency
    false, //shaderResourceMinLod
    false, //sparseBinding
    false, //sparseResidencyBuffer
    false, //sparseResidencyImage2D
    false, //sparseResidencyImage3D
    false, //sparseResidency2Samples
    false, //sparseResidency4Samples
    false, //sparseResidency8Samples
    false, //sparseResidency16Samples
    false, //sparseResidencyAliased
    false, //variableMultisampleRate
    false, //inheritedQueries
};

VkPhysicalDeviceVulkan11Features requiredFeaturesExtended{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, //VkStructureType    sType;
    NULL, //void*              pNext;
    true, //VkBool32           storageBuffer16BitAccess;
    false, //VkBool32           uniformAndStorageBuffer16BitAccess;
    false, //VkBool32           storagePushConstant16;
    false, //VkBool32           storageInputOutput16;
    false, //VkBool32           multiview;
    false, //VkBool32           multiviewGeometryShader;
    false, //VkBool32           multiviewTessellationShader;
    true, //VkBool32           variablePointersStorageBuffer;
    true, //VkBool32           variablePointers;
    false, //VkBool32           protectedMemory;
    false, //VkBool32           samplerYcbcrConversion;
    false, //VkBool32           shaderDrawParameters;
};

// Provided by VK_VERSION_1_2
VkPhysicalDeviceVulkan12Features requiredFeaturesExtended2 {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, //VkStructureType    sType;
    &requiredFeaturesExtended, //void*              pNext;
    false, //VkBool32           samplerMirrorClampToEdge;
    false, //VkBool32           drawIndirectCount;
    true, //VkBool32           storageBuffer8BitAccess;
    false, //VkBool32           uniformAndStorageBuffer8BitAccess;
    false, //VkBool32           storagePushConstant8;
    false, //VkBool32           shaderBufferInt64Atomics;
    false, //VkBool32           shaderSharedInt64Atomics;
    fp16Support, //VkBool32           shaderFloat16;
    true, //VkBool32           shaderInt8;
    false, //VkBool32           descriptorIndexing;
    false, //VkBool32           shaderInputAttachmentArrayDynamicIndexing;
    false, //VkBool32           shaderUniformTexelBufferArrayDynamicIndexing;
    false, //VkBool32           shaderStorageTexelBufferArrayDynamicIndexing;
    false, //VkBool32           shaderUniformBufferArrayNonUniformIndexing;
    false, //VkBool32           shaderSampledImageArrayNonUniformIndexing;
    false, //VkBool32           shaderStorageBufferArrayNonUniformIndexing;
    false, //VkBool32           shaderStorageImageArrayNonUniformIndexing;
    false, //VkBool32           shaderInputAttachmentArrayNonUniformIndexing;
    false, //VkBool32           shaderUniformTexelBufferArrayNonUniformIndexing;
    false, //VkBool32           shaderStorageTexelBufferArrayNonUniformIndexing;
    false, //VkBool32           descriptorBindingUniformBufferUpdateAfterBind;
    false, //VkBool32           descriptorBindingSampledImageUpdateAfterBind;
    false, //VkBool32           descriptorBindingStorageImageUpdateAfterBind;
    false, //VkBool32           descriptorBindingStorageBufferUpdateAfterBind;
    false, //VkBool32           descriptorBindingUniformTexelBufferUpdateAfterBind;
    false, //VkBool32           descriptorBindingStorageTexelBufferUpdateAfterBind;
    false, //VkBool32           descriptorBindingUpdateUnusedWhilePending;
    false, //VkBool32           descriptorBindingPartiallyBound;
    false, //VkBool32           descriptorBindingVariableDescriptorCount;
    false, //VkBool32           runtimeDescriptorArray;
    false, //VkBool32           samplerFilterMinmax;
    false, //VkBool32           scalarBlockLayout;
    false, //VkBool32           imagelessFramebuffer;
    false, //VkBool32           uniformBufferStandardLayout;
    false, //VkBool32           shaderSubgroupExtendedTypes;
    false, //VkBool32           separateDepthStencilLayouts;
    false, //VkBool32           hostQueryReset;
    false, //VkBool32           timelineSemaphore;
    true, //VkBool32           bufferDeviceAddress;
    false, //VkBool32           bufferDeviceAddressCaptureReplay;
    false, //VkBool32           bufferDeviceAddressMultiDevice;
    false, //VkBool32           vulkanMemoryModel;
    false, //VkBool32           vulkanMemoryModelDeviceScope;
    false, //VkBool32           vulkanMemoryModelAvailabilityVisibilityChains;
    false, //VkBool32           shaderOutputViewportIndex;
    false, //VkBool32           shaderOutputLayer;
    false, //VkBool32           subgroupBroadcastDynamicId;
};

// Provided by VK_VERSION_1_3
VkPhysicalDeviceSynchronization2Features requiredFeaturesExtended3 {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    &requiredFeaturesExtended2,
    true
};

VkPhysicalDeviceFeatures2 requiredFeatures{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    &requiredFeaturesExtended3,
    requiredFeaturesBasic,
};

bool isDeviceSuitable(const VkPhysicalDevice& device) {
    if (!findQueueFamilies(device).indices.has_value()) return false;
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
    vkGetPhysicalDeviceFeatures2(device, &deviceFeatures);

    if (!deviceFeatures.features.shaderStorageBufferArrayDynamicIndexing) return false;
    //if (!deviceFeatures.features.shaderFloat64) return false;
    if (!deviceFeatures.features.shaderInt64) return false;
    if (!deviceFeatures.features.shaderInt16) return false;
    if (!deviceFeatures.features.logicOp) return false;
    //extension 3
    if (!extension3.synchronization2) return false;
    //extension 2
    if (!extension2.bufferDeviceAddress) return false;
    if (!extension2.shaderInt8) return false;
    if (fp16Support && !extension2.shaderFloat16) return false;
    //if (!extension2.uniformAndStorageBuffer8BitAccess) return false;
    if (!extension2.storageBuffer8BitAccess) return false;
    //if (!extension2.storagePushConstant8) return false;
    //extension 1
    if (!extension1.variablePointers) return false;
    if (!extension1.variablePointersStorageBuffer) return false;
    //if (!extension1.storageInputOutput16) return false; not supported in general
    //if (!extension1.storagePushConstant16) return false; useless
    //if (!extension1.uniformAndStorageBuffer16BitAccess) return false;
    if (!extension1.storageBuffer16BitAccess) return false;
    return true;
}

void beginCmdBuffer(VkCommandBuffer cmdbuff){
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(cmdbuff, &beginInfo) != VK_SUCCESS) throw std::runtime_error("Failed to End Command Buffer");
}
void endCmdBuffer(VkCommandBuffer cmdbuff){
    if (vkEndCommandBuffer(cmdbuff) != VK_SUCCESS) throw std::runtime_error("Failed to End Command Buffer");
}

void resetCmdBuffer(VkCommandBuffer cmdbuff){
    if (vkResetCommandBuffer(cmdbuff, 0) != VK_SUCCESS) throw std::runtime_error("Failed to reset Command Buffer");
}

class VkCommandPoolManager{
    VkDevice& device;
    VkCommandPool commandPool;
public:
    std::vector<VkCommandBuffer> primaryCommandBuffers;
    std::vector<VkCommandBuffer> secondaryCommandBuffers;
    VkCommandPoolManager(uint bufferNbPrimary, uint bufferNbSecondary, VkDevice& device, QueueFamily queueFamily) : device(device){
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamily.indices.value();
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("echec de la création d'une command pool!");
        }
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        if (bufferNbPrimary != 0){
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = bufferNbPrimary;
            primaryCommandBuffers.resize(bufferNbPrimary);
            if (vkAllocateCommandBuffers(device, &allocInfo, primaryCommandBuffers.data()) != VK_SUCCESS) {
                throw std::runtime_error("echec de l'allocation de command buffers!");
            }
        }
        if (bufferNbSecondary != 0){
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            allocInfo.commandBufferCount = bufferNbSecondary;
            secondaryCommandBuffers.resize(bufferNbSecondary);
            if (vkAllocateCommandBuffers(device, &allocInfo, secondaryCommandBuffers.data()) != VK_SUCCESS) {
                throw std::runtime_error("echec de l'allocation de command buffers!");
            }
        }
    }
    ~VkCommandPoolManager(){
        vkDestroyCommandPool(device, commandPool, nullptr);
    }
};

class FenceManager{
    VkDevice& device;
public:
    VkFence fence;
    FenceManager(VkDevice& device): device(device){
        VkFenceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        if (vkCreateFence(device, &createInfo, nullptr, &fence) != VK_SUCCESS) throw std::runtime_error("Failed to Create Fence");
    }
    //reset the fence right after
    void synchronize(){
        if (vkWaitForFences(device, 1, &fence, true, UINT64_MAX) != VK_SUCCESS) throw std::runtime_error("Failed to synchronize to Fence");
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS) throw std::runtime_error("Failed to reset Fence");
    }
    bool getStatus(){
        VkResult res = vkGetFenceStatus(device, fence);
        if (res == VK_ERROR_DEVICE_LOST) throw std::runtime_error("device lost while getting fence status");
        return (res == VK_SUCCESS);
    }
    ~FenceManager(){
        vkDestroyFence(device, fence, nullptr);
    }
};

struct MemoryBarrierPipeline{
    VkDeviceAddress address;
    size_t size;
    VkAccessFlags2 srcMask;
    VkAccessFlags2 dstMask;
};

class VkDeviceManager{
    VkPhysicalDeviceMemoryProperties memProps;
    VkDevice device;
    QueueFamily queueFamily;
    std::vector<VkQueue> deviceQueues;
    std::mutex* queueLocks;
    int queueID = 0; std::mutex queueIDLock;
    VkPipelineLayout pipelineLayout;
    //memory managment elemnts
    VmaAllocator vmaAllocator;
    std::map<VkDeviceAddress, std::pair<VmaAllocation, VkBuffer>, std::greater<VkDeviceAddress>> allocatorPerAdress;
    std::mutex allocatorPerAddressLock;
    std::map<const void*, std::pair<VkDeviceAddress, size_t>, std::greater<const void*>> hostMapToInfo;
    std::mutex hostMapToInfoLock;
    VkDeviceAddress getBufferDevicePointer(VkBuffer& buffer){
        VkDeviceAddress res;
        VkBufferDeviceAddressInfo data;
        data.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        data.buffer = buffer;
        data.pNext = NULL;
        res = vkGetBufferDeviceAddress(device, &data);
        if (res == 0) throw std::runtime_error("Adress of Buffer is NULL!");
        return res;
    }
    void createLogicalDevice(VkPhysicalDevice& physicalDevice){
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily.indices.value();
        queueCreateInfo.queueCount = queueFamily.maxHWQueueCount;
        std::vector<float> queuePriorities(queueFamily.maxHWQueueCount, 1.f);
        queueCreateInfo.pQueuePriorities = queuePriorities.data();

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.queueCreateInfoCount = 1;

        createInfo.pNext = &requiredFeatures;
        createInfo.pEnabledFeatures = nullptr;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }
        deviceQueues.resize(queueFamily.maxHWQueueCount);
        for (int i = 0; i < queueFamily.maxHWQueueCount; i++){
            vkGetDeviceQueue(device, queueFamily.indices.value(), i, deviceQueues.data()+i);
        }
        queueLocks = new std::mutex[queueFamily.maxHWQueueCount];
    }
    void initVmaAllocator(VkInstance& instance, VkPhysicalDevice& physicalDevice){
        VmaAllocatorCreateInfo createInfo{};
        createInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        createInfo.physicalDevice = physicalDevice;
        createInfo.device = device;
        createInfo.instance = instance;
        if (vmaCreateAllocator(&createInfo, &vmaAllocator) != VK_SUCCESS) throw std::runtime_error("Failed to initialize VmaAllocator");
    }
    void initPipelineLayout(){
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutCreateInfo.pNext = NULL;

        //we don't use descriptors
        pipelineLayoutCreateInfo.setLayoutCount = 0;
        pipelineLayoutCreateInfo.pSetLayouts = NULL;
        //we use push constants to pass our arguments (up to 16 arguments of size 8 bytes -> 128B supported everywhere)
        VkPushConstantRange constantRangeInfo{};
        constantRangeInfo.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        constantRangeInfo.offset = 0;
        constantRangeInfo.size = 128;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &constantRangeInfo;
        if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS){
            throw std::runtime_error("Failed to create Pipeline Layout");
        }
    }
    std::pair<VkDeviceAddress, VmaAllocation> malloc(size_t size, VkBufferUsageFlags bufferFlags, VmaMemoryUsage vmaAllocationUsage, VkMemoryPropertyFlags allocationFlags, VmaAllocationCreateFlags allocInfoFlags){
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.pNext = NULL;
        bufferInfo.size = size;
        bufferInfo.usage = bufferFlags | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
 
        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = vmaAllocationUsage;
        allocInfo.requiredFlags = allocationFlags;
        allocInfo.flags = allocInfoFlags;
 
        VkBuffer buffer;
        VmaAllocation allocation;
        if (vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) throw std::runtime_error("Failed to Allocate Memory");
        VkDeviceAddress address = getBufferDevicePointer(buffer);
        allocatorPerAddressLock.lock();
        allocatorPerAdress.emplace(address, std::make_pair(allocation, buffer));
        allocatorPerAddressLock.unlock();
        return std::make_pair(address, allocation);
    }
    //no boundary check
    VkBuffer getMemoryBuffer(VkDeviceAddress address){
        allocatorPerAddressLock.lock();
        auto it = allocatorPerAdress.lower_bound(address);
        if (it == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto res = it->second.second;
        allocatorPerAddressLock.unlock();
        return res;
    }
public:
    VkDeviceManager(VkInstance& instance, VkPhysicalDevice& physicalDevice){
        queueFamily = findQueueFamilies(physicalDevice);
        createLogicalDevice(physicalDevice);
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        initVmaAllocator(instance, physicalDevice);
        initPipelineLayout();
    }
    VkCommandPoolManager* createCmdPool(uint bufferNbPrimary, uint bufferNbSecondary){
        return new VkCommandPoolManager(bufferNbPrimary, bufferNbSecondary, device, queueFamily);
    }
    VkShaderModule loadShaderModule(std::string filename){
        //reading file
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Could not open Shader file");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> bytecode(fileSize);
        file.seekg(0);
        file.read(bytecode.data(), fileSize);
        file.close();

        //creating shadermodule
        return loadShaderModule(bytecode);
    }
    VkShaderModule loadShaderModule(std::vector<char>& bytecode){

        //creating shadermodule
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = bytecode.size();
        createInfo.pCode = (uint32_t*)(bytecode.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }
    void freeShaderModule(VkShaderModule shaderModule){
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
    VkPipeline createComputePipeline(VkShaderModule shaderModule, std::string entry_point){
        VkComputePipelineCreateInfo pipelineCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineCreateInfo.stage.module = shaderModule;
        pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineCreateInfo.stage.pName = entry_point.c_str();
        pipelineCreateInfo.layout = pipelineLayout;
        pipelineCreateInfo.pNext = NULL;

        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult err = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);
        if (err != VK_SUCCESS){
            throw std::runtime_error("Error " + std::to_string(err) + " while creating compute pipeline");
        }
        return pipeline;
    }
    void destroyComputePipeline(VkPipeline pipeline){
        vkDestroyPipeline(device, pipeline, nullptr);
    }
    void* mallocHost(size_t size){
        auto [address, allocation] = malloc(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO, 0, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* res;
        vmaMapMemory(vmaAllocator, allocation, &res);
        hostMapToInfoLock.lock();
        hostMapToInfo.emplace((const void*)res, std::make_pair(address, size));
        hostMapToInfoLock.unlock();
        return res;
    }
    VkDeviceAddress mallocDevice(size_t size){
        return malloc(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0).first;
    }
    void freeDevice(VkDeviceAddress address){
        allocatorPerAddressLock.lock();
        if (allocatorPerAdress.find(address) == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("Device free an address that isnt allocated");
        }
        auto [allocation, buffer] = allocatorPerAdress[address];
        allocatorPerAdress.erase(address);
        allocatorPerAddressLock.unlock();
        vkDestroyBuffer(device, buffer, NULL);
        vmaFreeMemory(vmaAllocator, allocation);
    }
    //works even if ptr isnt the original. As long as ptr is in range of host allocated memory it's good
    bool isHostAllocated(const void* ptr){
        try{
            (void)getHostMemoryDeviceAddress(ptr);
            return true;
        } catch(std::runtime_error& e){
            return false;
        }
    }
    //returns the correct offset deviceAddress
    VkDeviceAddress getHostMemoryDeviceAddress(const void* ptr){
        //map is in greater mode so lower_bound is first <= element
        hostMapToInfoLock.lock();
        auto it = hostMapToInfo.lower_bound(ptr);
        if (it == hostMapToInfo.end()) {
            hostMapToInfoLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto [allocationPtr, data] = *it;
        hostMapToInfoLock.unlock();
        auto [address, size] = data;
        //we have (allocationPtr <= ptr)
        if ((uint8_t*)allocationPtr+size <= (uint8_t*)ptr) throw std::runtime_error("unmapped pointer");
        return address+((uint8_t*)ptr - (uint8_t*)allocationPtr);
    }
    //must be the exact ptr
    void freeHost(void* ptr){
        hostMapToInfoLock.lock();
        if (hostMapToInfo.find(ptr) == hostMapToInfo.end()) {
            hostMapToInfoLock.unlock();
            throw std::runtime_error("Freeing a ptr that wasnt allocated");
        }
        auto [address, size] = hostMapToInfo[ptr];
        hostMapToInfo.erase(ptr);
        hostMapToInfoLock.unlock();
        allocatorPerAddressLock.lock();
        auto [allocation, buffer] = allocatorPerAdress[address];
        allocatorPerAdress.erase(address);
        allocatorPerAddressLock.unlock();
        vmaUnmapMemory(vmaAllocator, allocation);
        vkDestroyBuffer(device, buffer, NULL);
        vmaFreeMemory(vmaAllocator, allocation);
    }
    //will place the copy command in cmdbuff (be careful to have cmdbuff to a recording state)
    //no boundary check
    void memCpyHtoD(VkCommandBuffer cmdbuff, VkDeviceAddress dst, const void* src, size_t size){
        hostMapToInfoLock.lock();
        auto itSrc = hostMapToInfo.lower_bound(src);
        if (itSrc == hostMapToInfo.end()) {
            hostMapToInfoLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto [allocationPtrSrc, deviceAddressSrc] = *itSrc;
        hostMapToInfoLock.unlock();
        allocatorPerAddressLock.lock();
        auto itDst = allocatorPerAdress.lower_bound(dst);
        if (itDst == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto bufferSrc = allocatorPerAdress[deviceAddressSrc.first].second;
        allocatorPerAddressLock.unlock();
        auto& [allocationAddressDst, bufferAndAllocatorDst] = *itDst;
        VkBufferCopy copyInfo;
        copyInfo.size = size;
        copyInfo.srcOffset = (uint8_t*)src - (uint8_t*)allocationPtrSrc;
        copyInfo.dstOffset = dst-allocationAddressDst;
        vkCmdCopyBuffer(cmdbuff, bufferSrc, bufferAndAllocatorDst.second, 1, &copyInfo);
    }
    //will place the copy command in cmdbuff (be careful to have cmdbuff to a recording state)
    //no boundary check
    void memCpyDtoH(VkCommandBuffer cmdbuff, void* dst, VkDeviceAddress src, size_t size){
        hostMapToInfoLock.lock();
        auto itDst = hostMapToInfo.lower_bound(dst);
        if (itDst == hostMapToInfo.end()) {
            hostMapToInfoLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto [allocationPtrDst, deviceAddressDst] = *itDst;
        hostMapToInfoLock.unlock();
        allocatorPerAddressLock.lock();
        auto itSrc = allocatorPerAdress.lower_bound(src);
        if (itSrc == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto& bufferDst = allocatorPerAdress[deviceAddressDst.first].second;
        auto& [allocationAddressSrc, bufferAndAllocatorSrc] = *itSrc;
        allocatorPerAddressLock.unlock();
        VkBufferCopy copyInfo;
        copyInfo.size = size;
        copyInfo.srcOffset = src-allocationAddressSrc;
        copyInfo.dstOffset = (uint8_t*)dst - (uint8_t*)allocationPtrDst;
        vkCmdCopyBuffer(cmdbuff, bufferAndAllocatorSrc.second, bufferDst, 1, &copyInfo);
    }
    //will place the copy command in cmdbuff (be careful to have cmdbuff to a recording state)
    //no boundary check
    void memCpyDtoD(VkCommandBuffer cmdbuff, VkDeviceAddress dst, VkDeviceAddress src, size_t size){
        allocatorPerAddressLock.lock();
        auto itDst = allocatorPerAdress.lower_bound(dst);
        if (itDst == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto itSrc = allocatorPerAdress.lower_bound(src);
        if (itSrc == allocatorPerAdress.end()) {
            allocatorPerAddressLock.unlock();
            throw std::runtime_error("unmapped pointer");
        }
        auto [allocationAddressDst, bufferAndAllocatorDst] = *itDst;
        auto [allocationAddressSrc, bufferAndAllocatorSrc] = *itSrc;
        allocatorPerAddressLock.unlock();
        VkBufferCopy copyInfo;
        copyInfo.size = size;
        copyInfo.srcOffset = src-allocationAddressSrc;
        copyInfo.dstOffset = dst-allocationAddressDst;
        vkCmdCopyBuffer(cmdbuff, bufferAndAllocatorSrc.second, bufferAndAllocatorDst.second, 1, &copyInfo);
    }
    //VkAccessFlag2, you can use VK_ACCESS_2_MEMORY_WRITE_BIT and VK_ACCESS_2_MEMORY_READ_BIT
    void placeMemoryBarriers(VkCommandBuffer cmdbuff, std::vector<MemoryBarrierPipeline> barriers){
        std::vector<VkBufferMemoryBarrier2> barriersInfos(barriers.size());
        
        for (int i = 0; i < barriers.size(); i++){
            VkDeviceAddress address = barriers[i].address;
            size_t size = barriers[i].size;
            VkAccessFlags2 srcMask = barriers[i].srcMask;
            VkAccessFlags2 dstMask = barriers[i].dstMask;
            //obtain buffer and offset and size
            allocatorPerAddressLock.lock();
            auto it = allocatorPerAdress.lower_bound(address);
            if (it == allocatorPerAdress.end()) {
                allocatorPerAddressLock.unlock();
                throw std::runtime_error("unmapped pointer");
            }
            auto [allocationAddress, bufferAndAllocator] = *it;
            allocatorPerAddressLock.unlock();

            VkBufferMemoryBarrier2& createInfo = barriersInfos[i];
            createInfo.sType =  VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            createInfo.pNext = NULL;
            createInfo.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            createInfo.dstStageMask = createInfo.srcStageMask;
            createInfo.srcAccessMask = srcMask;
            createInfo.dstAccessMask = dstMask;
            createInfo.srcQueueFamilyIndex = queueFamily.indices.value();
            createInfo.dstQueueFamilyIndex = queueFamily.indices.value();
            createInfo.buffer = bufferAndAllocator.second;
            createInfo.offset = address-allocationAddress;
            createInfo.size = size;
        }

        VkDependencyInfo createInfo2{};
        createInfo2.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        createInfo2.pNext = NULL;
        createInfo2.pImageMemoryBarriers = NULL;
        createInfo2.pMemoryBarriers = NULL;
        createInfo2.pBufferMemoryBarriers = barriersInfos.data();
        createInfo2.imageMemoryBarrierCount = 0;
        createInfo2.memoryBarrierCount = 0;
        createInfo2.bufferMemoryBarrierCount = barriersInfos.size();

        vkCmdPipelineBarrier2(cmdbuff, &createInfo2);
    }
    //arguments should be 128 bytes allocated
    void placeKernel(VkCommandBuffer cmdbuff, VkPipeline computePipeline, uint8_t* arguments, uint64_t bl_x, uint64_t bl_y = 1, uint64_t bl_z = 1){
        vkCmdPushConstants(cmdbuff, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, arguments);
        vkCmdBindPipeline(cmdbuff, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdDispatch(cmdbuff, bl_x, bl_y, bl_z);
    }
    FenceManager* createFence(){
        return new FenceManager(device);
    }
    VkSemaphore createSemaphores(){
        VkSemaphore res;
        VkSemaphoreCreateInfo createInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(device, &createInfo, nullptr, &res) != VK_SUCCESS) throw std::runtime_error("Failed to create semaphore");
        return res;
    }
    void destroySemaphores(VkSemaphore a){
        vkDestroySemaphore(device, a, nullptr);
    }
    void executeBuffer(VkCommandBuffer cmdbuff, FenceManager* fence){
        queueIDLock.lock();
        int myQueueID = queueID;
        queueID = (queueID+1)%deviceQueues.size();
        queueIDLock.unlock();

        VkSubmitInfo2 createInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        createInfo.pNext = NULL;
        createInfo.flags = 0;
        createInfo.waitSemaphoreInfoCount = 0;
        createInfo.pWaitSemaphoreInfos = NULL;
        createInfo.signalSemaphoreInfoCount = 0;
        createInfo.pSignalSemaphoreInfos = NULL;

        VkCommandBufferSubmitInfo cmdBufferInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmdBufferInfo.pNext = NULL;
        cmdBufferInfo.commandBuffer = cmdbuff;
        cmdBufferInfo.deviceMask = 0; //corresponds to everyone
        createInfo.commandBufferInfoCount = 1;
        createInfo.pCommandBufferInfos = &cmdBufferInfo;

        queueLocks[myQueueID].lock();
        if (vkQueueSubmit2(deviceQueues[myQueueID], 1, &createInfo, fence->fence) != VK_SUCCESS) {
            queueLocks[myQueueID].unlock();
            throw std::runtime_error("Failed to Submit commandBuffer to queue");
        }
        queueLocks[myQueueID].unlock();
    }
    //execute all given cmdbuffs in order with the provided numBuff-1 semaphores
    void executeBuffers(VkCommandBuffer* cmdbuffs, VkSemaphore* semaphores, uint numBuff, FenceManager* fence){
        queueIDLock.lock();
        int myQueueID = queueID;
        queueID = (queueID+1)%deviceQueues.size();
        queueIDLock.unlock();

        std::vector<VkSubmitInfo2> createInfos(numBuff);
        std::vector<VkCommandBufferSubmitInfo> createInfosBuffers(numBuff);
        std::vector<VkSemaphoreSubmitInfo> createInfosSemaphore(numBuff-1);
        for (uint i = 0; i < numBuff-1; i++){
            auto& createInfo = createInfosSemaphore[i];
            createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            createInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            createInfo.pNext = NULL;
            createInfo.deviceIndex = 0;
            createInfo.value = 0;
            createInfo.semaphore = semaphores[i];
        }
        for (uint i = 0; i < numBuff; i++){
            auto& createInfo = createInfos[i];
            auto& createInfoBuffer = createInfosBuffers[i];
            createInfoBuffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            createInfoBuffer.pNext = NULL;
            createInfoBuffer.deviceMask = 0;
            createInfoBuffer.commandBuffer = cmdbuffs[i];
            createInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            createInfo.pNext = NULL;
            createInfo.flags = 0;
            createInfo.waitSemaphoreInfoCount = (i == 0) ? 0 : 1;
            createInfo.pWaitSemaphoreInfos = (i==0) ? NULL : &createInfosSemaphore[i-1];
            createInfo.signalSemaphoreInfoCount = (i == numBuff-1) ? 0: 1;
            createInfo.pSignalSemaphoreInfos = (i == numBuff-1) ? NULL : &createInfosSemaphore[i];
            createInfo.commandBufferInfoCount = 1;
            createInfo.pCommandBufferInfos = createInfosBuffers.data()+i;
        }

        queueLocks[myQueueID].lock();
        if (vkQueueSubmit2(deviceQueues[myQueueID], numBuff, createInfos.data(), fence->fence) != VK_SUCCESS) {
            queueLocks[myQueueID].unlock();
            throw std::runtime_error("Failed to Submit commandBuffer to queue");
        }
        queueLocks[myQueueID].unlock();
    }
    ~VkDeviceManager(){
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vmaDestroyAllocator(vmaAllocator);
        delete[] queueLocks;
        vkDestroyDevice(device, nullptr);
    }
};

void fillKernelArguments(int current_index, uint8_t* array){
    return;
}

//helper to fill arguments
template <typename T, typename... Types>
void fillKernelArguments(int current_index, uint8_t* array, T arg, Types... variadicArgs){
    //manage alignment which is in byte
    current_index = (current_index+sizeof(T)-1)/sizeof(T)*sizeof(T);
    *(T*)(array+current_index) = arg;
    current_index += sizeof(T);
    fillKernelArguments(current_index, array, variadicArgs...);
}

//possesses its instance
class VkManager{
    VkDebugUtilsMessengerEXT debugMessenger;
    void initInstance(){
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vship";
        appInfo.applicationVersion = VK_MAKE_VERSION(5, 0, 0);
        appInfo.pEngineName = "VshipEngine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        //manage extensions
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        createInfo.enabledExtensionCount = (uint32_t)required_extensions.size();
        createInfo.ppEnabledExtensionNames = required_extensions.data();

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create vulkan instance!");
        }
    }
    void setupDebugMessenger() {
        if (!enableValidationLayers) return;
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }
public:
    VkInstance instance;
    VkManager(){
        if (enableValidationLayers && !checkValidationLayerSupport())throw std::runtime_error("Validation Layer Not Supported");
        initInstance();
        setupDebugMessenger();
    }
    //raw disables the check for isDeviceSuitable to get a raw list. TO NOT USE for VkDeviceManager
    std::vector<VkPhysicalDevice> listGPU(bool raw = false){
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            return {};
            //throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        std::vector<VkPhysicalDevice> out;
        for (const auto& el: devices){
            if (raw || isDeviceSuitable(el)){
                out.push_back(el);
            }
        }
        return out;
    }

    ~VkManager(){
        if (enableValidationLayers) DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};