#pragma once

namespace cvvdp{

//Inverse of a real FFT of length size (like numpy/torch irfft) followed by an fftshift, from its first
//inp.size() coefficients; the missing ones count as 0.
std::vector<float> inverseRealFourrierTransform(std::vector<float> inp, uint size){
    //every coefficient but the DC one, and the Nyquist one when size is even, stands for itself and its mirror
    if (size%2 == 0 && inp.size() > size/2){
        inp[size/2] /= 2;
    }
    inp[0] /= 2;

    std::vector<float> res(size, 0);
    for (uint i = 0; i < size; i++){
        for (uint k = 0; k < inp.size(); k++){
            //we offset the index of res to get frequencies starting from 0 then positive then negative
            res[(i+inp.size()-1)%size] += 2*inp[k]*std::cos(2*PI*k*i/size)/size; //2/N is normalization factor and exceptions are handled before
        }
    }

    return res;
}

class TemporalFilter{
    VshipDeviceHelper* deviceHelper;
    float* cpumem = NULL;
public:
    VkDeviceAddress convolutionKernel_d = 0; //already prepared for GPU
    int size = 0;
    TemporalFilter(){
    }
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff, float fps){
        this->deviceHelper = deviceHelper;
        size = (ceil(0.25*fps/2)*2)+1;
        size = std::max(size, 1);

        convolutionKernel_d = deviceHelper->device.mallocDevice(sizeof(float)*size*4);
        cpumem  = (float*)deviceHelper->device.mallocHost(sizeof(float)*size*4);

        int fourrier_size = size/2 + 1; //only uses real FT
        
        std::vector<float> freq(fourrier_size);
        //the frequencies of the real FFT's coefficients, 0 to fps/2 (pycvvdp's linspace(0, fps/2, fourrier_size))
        for (int i = 0; i < fourrier_size; i++) freq[i] = (fourrier_size > 1) ? (fps/2)*i/(fourrier_size-1) : 0.f;

        //Y sustained, rg sustained, yv sustained, Y transient
        std::vector<float> filters_fftdomain[4];
        //sustained channels
        for (int j = 0; j < 3; j++){
            filters_fftdomain[j].resize(fourrier_size);
            for (int i = 0; i < fourrier_size; i++){
                filters_fftdomain[j][i] = std::exp(-std::pow(freq[i], beta_tf[j])/sigma_tf[j]);
            }
        }
        //transient
        filters_fftdomain[3].resize(fourrier_size);
        for (int i = 0; i < fourrier_size; i++){
            //5 is the peak frequency of the transient which is a pass band
            float temp = std::pow(freq[i], beta_tf[3]) - std::pow(5, beta_tf[3]);
            temp *= temp; //square
            filters_fftdomain[3][i] = std::exp(-temp/sigma_tf[3]);
        }

        //go back from the FFT world, this will become a convolution kernel instead of a frequency multiplicator
        std::vector<float> temporal_filters[4];
        for (int j = 0; j < 4; j++) temporal_filters[j] = inverseRealFourrierTransform(filters_fftdomain[j], size);

        for (int j = 0; j < 4; j++){
            for (int i = 0; i < size; i++){
                cpumem[j*size + i] = temporal_filters[j][i];
            }
        }
        deviceHelper->device.memCpyHtoD(cmdbuff, convolutionKernel_d, cpumem, sizeof(float)*4*size);
    }
    VkDeviceAddress getFilter(int j){ //returns gpu adresses
        //j must be 0, 1, 2 or 3. If not the pointer will be invalid
        //the size of the returned array is size
        return convolutionKernel_d + sizeof(float)*j*size;
    }
    void destroy(){
        deviceHelper->device.freeDevice(convolutionKernel_d);
        deviceHelper->device.freeHost(cpumem);
    }
};

//will contain the temporal image frame and manage them. 
// It will be able to apply the temporal filter to write 4 planes that corresponds to the 4 temporally filtered channels
class TemporalRing{
    VshipDeviceHelper* deviceHelper;
    //to get index of frame i, you need to do internal_memory+(3*plane_size)*((ind0+i)%max_temporal_size)
    //3* is because the 3 color planes of a single plane are stored together.
public:
    VkDeviceAddress internal_memory_d = 0;
    TemporalFilter tempFilterPreprocessor;
    int ind0 = 0; //index of current frame, past frames are stored after it up to temporal_size frames.
    int temporal_size = 0;
    int max_temporal_size = 0;
    int64_t width = -1;
    int64_t height = -1;
    TemporalRing(){}
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff, float fps, int64_t width, int64_t height){
        this->deviceHelper = deviceHelper;
        this->width = width;
        this->height = height;
        tempFilterPreprocessor.init(deviceHelper, cmdbuff, fps);
        const int num_channel = 3;
        max_temporal_size = tempFilterPreprocessor.size;
        //std::cout << "allocationBytes temporalRing : " << sizeof(float)*width*height*max_temporal_size*num_channel << " for temporal buffer of size " << max_temporal_size << std::endl;
        internal_memory_d = deviceHelper->device.mallocDevice(sizeof(float)*width*height*max_temporal_size*num_channel);
        if (internal_memory_d == 0){
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }
    }
    //the pointer has all 3 planes next to each others
    //i is "how old" is the frame. 0 is current, 1 is previous,...
    VkDeviceAddress getFramePointer(int i){
        assert(temporal_size > 0);
        //if i asked is too old, CVVDP goes to the oldest frame in store
        if (i >= temporal_size) i = temporal_size-1;
        return internal_memory_d+sizeof(float)*(3*width*height)*((ind0+i)%max_temporal_size);
    }
    //the ring suppose the current frame has been written
    //after rotation, frame 0 needs to be written too to be valid
    //this is done using getFramePointer(0) and writing the frame on the pointer
    void rotate(){
        //we increment temporal_size to signify we have one more frame in store now
        //if we reached max, it means the oldest one will be overwritten (intended)
        temporal_size = std::min(max_temporal_size, temporal_size+1);
        ind0 = (ind0 + max_temporal_size - 1)%max_temporal_size; //force maintain positive modulo
    }
    void reset(){
        temporal_size = 0;
    }
    void destroy(){
        deviceHelper->device.freeDevice(internal_memory_d);
        tempFilterPreprocessor.destroy();
    }
};

//give it planes, it will overwrite with the 4 temporal channels
void placeComputeTemporalChannels(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdBuff, TemporalRing& ring, VkDeviceAddress Y_sustained, VkDeviceAddress RG_sustained, VkDeviceAddress YV_sustained, VkDeviceAddress Y_transient){
    int th_x = 256;
    int bl_x = (ring.width*ring.height+th_x-1)/th_x;

    VkPipeline pipeline = deviceHelper->cachedPipeline["temporalFilterCVVDP.spv:temporalConvolutionKernel_d"];
    uint8_t kernelArguments[128];
    fillKernelArguments(0, kernelArguments, Y_sustained, RG_sustained, YV_sustained, Y_transient, ring.width, ring.height, ring.tempFilterPreprocessor.convolutionKernel_d, ring.internal_memory_d, ring.ind0, ring.temporal_size, ring.max_temporal_size);
    deviceHelper->device.placeKernel(cmdBuff, pipeline, kernelArguments, bl_x, 1, 1);
    deviceHelper->device.placeMemoryBarriers(cmdBuff, {
        {Y_sustained, sizeof(float)*ring.width*ring.height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {RG_sustained, sizeof(float)*ring.width*ring.height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {YV_sustained, sizeof(float)*ring.width*ring.height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {Y_transient, sizeof(float)*ring.width*ring.height, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_MEMORY_READ_BIT},
        {ring.internal_memory_d, sizeof(float)*ring.width*ring.height*ring.max_temporal_size, VK_ACCESS_2_MEMORY_READ_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT}
    });
}


}