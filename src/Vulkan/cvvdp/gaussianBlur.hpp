#pragma once

namespace cvvdp{

class GaussianHandle{
public:
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress gaussiankernel_d = 0;
    VkDeviceAddress gaussiankernel_integral_d = 0;
    float* gaussiankernel;
    //this cmdbuff needs to be executed as pretreatment at init  
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff){
        this->deviceHelper = deviceHelper;
        gaussiankernel = (float*)deviceHelper->device.mallocHost((4*GAUSSIANSIZE+3)*sizeof(float));
        gaussiankernel[2*GAUSSIANSIZE+1] = 0; //will be integral
        for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
            gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*pu_dilate*pu_dilate))/(std::sqrt(TAU*pu_dilate*pu_dilate));
            gaussiankernel[2*GAUSSIANSIZE+2+i] = gaussiankernel[2*GAUSSIANSIZE+1+i] + gaussiankernel[i];
        }
        gaussiankernel_d = deviceHelper->device.mallocDevice(sizeof(float)*(4*GAUSSIANSIZE+3));
        deviceHelper->device.memCpyHtoD(cmdbuff, gaussiankernel_d, gaussiankernel, (4*GAUSSIANSIZE+3)*sizeof(float));
        gaussiankernel_integral_d = gaussiankernel_d + sizeof(float)*(2*GAUSSIANSIZE+1);
    }
    void destroy(){
        deviceHelper->device.freeHost(gaussiankernel);
        deviceHelper->device.freeDevice(gaussiankernel_d);
    }
};

}