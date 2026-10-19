#pragma once

namespace cvvdp{

//pycvvdp's phase uncertainty blur is torchvision's GaussianBlur(int(4*pu_dilate)+1, pu_dilate): a 13-tap Gaussian
//of sigma 3 normalised to a sum of 1, with reflected borders (shaders/maskingModel.slang).
constexpr int puRadius = 2*pu_dilate;

class GaussianHandle{
public:
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress gaussiankernel_d = 0;
    float* gaussiankernel;
    //this cmdbuff needs to be executed as pretreatment at init
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff){
        this->deviceHelper = deviceHelper;
        const int taps = 2*puRadius+1;
        gaussiankernel = (float*)deviceHelper->device.mallocHost(taps*sizeof(float));
        //computed in float like torchvision's _get_gaussian_kernel1d
        float sum = 0.f;
        for (int i = 0; i < taps; i++){
            const float x = (float)(i - puRadius)/(float)pu_dilate;
            gaussiankernel[i] = std::exp(-0.5f*x*x);
            sum += gaussiankernel[i];
        }
        for (int i = 0; i < taps; i++) gaussiankernel[i] /= sum;
        gaussiankernel_d = deviceHelper->device.mallocDevice(sizeof(float)*taps);
        deviceHelper->device.memCpyHtoD(cmdbuff, gaussiankernel_d, gaussiankernel, taps*sizeof(float));
    }
    void destroy(){
        deviceHelper->device.freeHost(gaussiankernel);
        deviceHelper->device.freeDevice(gaussiankernel_d);
    }
};

}
