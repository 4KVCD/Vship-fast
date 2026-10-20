namespace butter{

//libjxl's ComputeKernel (lib/jxl/butteraugli/butteraugli.cc), the kernel the CPU Butteraugli blurs with: radius
//max(1, int(2.25*sigma)), weights exp(-i*i/(2*sigma*sigma)) without normalisation. Writes the 2*radius+1 weights,
//then weights[0..radius] times 1/(sum of the weights), as libjxl's blurs scale them (both ConvolutionWithTranspose
//and Separable5 sum the weights in float in increasing order). Returns the radius.
inline int loadGaussianKernel(float* out, float sigma){
    const double scaler = -1.0 / (2.0 * sigma * sigma);
    const float m = 2.25f;
    const int radius = std::max<int>(1, m * std::fabs(sigma));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++){
        out[i + radius] = std::exp(scaler * i * i);
        sum += out[i + radius];
    }
    const float scale = 1.0f / sum;
    for (int i = 0; i <= radius; i++){
        out[2*radius + 1 + i] = out[i] * scale;
    }
    return radius;
}

//Butteraugli's 5 kernels on the GPU: i = 0 to 4 for the sigmas 1.2, 1.564, 2.7, 3.22 and 7.156 (radii 2, 3, 6, 7
//and 16), each in the layout loadGaussianKernel writes.
class GaussianHandle{
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress gaussiankernel_d;
    float* staging_buffer;
    int indices[5];
    int radii[5];
public:
    //cmdBuff is preprocessing buffer
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff){
        this->deviceHelper = deviceHelper;
        gaussiankernel_d = deviceHelper->device.mallocDevice(sizeof(float)*1024);
        staging_buffer = (float*)deviceHelper->device.mallocHost(sizeof(float)*1024);

        //libjxl's sigmas, as floats like its Blur's argument
        const float sigmas[5] = {1.2f, 1.56416327805f, 2.7f, 3.22489901262f, 7.15593339443f};
        int next = 0;
        for (int i = 0; i < 5; i++){
            indices[i] = next;
            radii[i] = loadGaussianKernel(staging_buffer + next, sigmas[i]);
            next += 3*radii[i] + 2;
        }
        deviceHelper->device.memCpyHtoD(cmdBuff, gaussiankernel_d, staging_buffer, (1024*sizeof(float)));
    }
    VkDeviceAddress get(int i){
        return gaussiankernel_d+sizeof(float)*indices[i];
    }
    int getRadius(int i){
        return radii[i];
    }
    void destroy(){
        deviceHelper->device.freeDevice(gaussiankernel_d);
        deviceHelper->device.freeHost(staging_buffer);
    }
};

//The passes placeBlur can do after the blur, writing dst and second (postOp in separableBlur.slang; second must not
//be mem_d)
enum BlurPostOp {
    BLUR_POST_NONE = 0,
    BLUR_POST_RESIDUAL = 1,         //second = mem_d - blur
    BLUR_POST_REMOVE_RANGES = 2,    //subarray_removerangearound0 (1.5) then removerangearound0 (0.04) on second
    BLUR_POST_SEPARATE_HF_UHF = 3,  //separateHf_Uhf of the blur and mem_d
    BLUR_POST_SUBTRACT_REMOVE_RANGE = 4,  //second = mem_d - blur, then dst = RemoveRangeAroundZero(blur, w)
    BLUR_POST_SUBTRACT_AMPLIFY_RANGE = 5  //second = mem_d - blur, then dst = AmplifyRangeAroundZero(blur, w)
};

//dst = blur i of mem_d (see GaussianHandle), both passes in one kernel (separableBlur.slang, 32x64 outputs per
//workgroup), then the pass postOp. temp is not needed: the kernel does not go through memory between its passes.
void placeBlur(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, VkDeviceAddress dst, VkDeviceAddress mem_d, VkDeviceAddress temp, uint64_t width, uint64_t height, GaussianHandle& gaussianHandle, int i, VkDeviceAddress second = 0, BlurPostOp postOp = BLUR_POST_NONE, float w = 0.0f){
    (void)temp;
    const std::string radius = std::to_string(gaussianHandle.getRadius(i));
    VkPipeline pipeline = deviceHelper->cachedPipeline["separableBlur" + radius + "Butter.spv:separableBlur" + radius];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, mem_d, dst, width, height, gaussianHandle.get(i), second, (int)postOp, w);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, (width-1)/32+1, (height-1)/64+1, 1);
    std::vector<MemoryBarrierPipeline> barriers = {
        {mem_d, width*height*sizeof(float), VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT},
        {dst, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT}
    };
    if (postOp != BLUR_POST_NONE) barriers.push_back({second, width*height*sizeof(float), VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT});
    deviceHelper->device.placeMemoryBarriers(cmdBuff, barriers);
}

}
