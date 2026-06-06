namespace ssimu2{

//these parameters needs to be the same in the shader "scoreSSIMU2" slang file

//for now only accepts squares Tiling (see GaussianSmart_device out and tot arrays)
//only allows powers of 2 for Thx and Thy due to pointer jumping (8/8 16/16 and 32/32)
constexpr int gaussianThx = 16;
constexpr int gaussianThy = 16;
//Optimization Island in gaussianSize lies in gaussianTh/2
constexpr int gaussianSize = 8;

constexpr int gaussianTileX = 1;
constexpr int gaussianTileY = 1;

constexpr int tamponWidth = (gaussianThx*gaussianTileX + 2*gaussianSize);
constexpr int tamponHeight = (gaussianThy*gaussianTileY + 2*gaussianSize);

class GaussianHandle{
public:
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress gaussiankernel_d = 0;
    VkDeviceAddress gaussiankernel_integral_d = 0;
    float* gaussiankernel;
    //this cmdbuff needs to be executed as pretreatment at init  
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff){
        this->deviceHelper = deviceHelper;
        gaussiankernel = (float*)deviceHelper->device.mallocHost((4*gaussianSize+3)*sizeof(float));
        gaussiankernel[2*gaussianSize+1] = 0; //will be integral
        for (int i = 0; i < 2*gaussianSize+1; i++){
            gaussiankernel[i] = std::exp(-(gaussianSize-i)*(gaussianSize-i)/(2*SIGMA*SIGMA))/(std::sqrt(TAU*SIGMA*SIGMA));
            gaussiankernel[2*gaussianSize+2+i] = gaussiankernel[2*gaussianSize+1+i] + gaussiankernel[i];
        }
        gaussiankernel_d = deviceHelper->device.mallocDevice(sizeof(float)*(4*gaussianSize+3));
        deviceHelper->device.memCpyHtoD(cmdbuff, gaussiankernel_d, gaussiankernel, (4*gaussianSize+3)*sizeof(float));
        gaussiankernel_integral_d = gaussiankernel_d + sizeof(float)*(2*gaussianSize+1);
    }
    void destroy(){
        deviceHelper->device.freeHost(gaussiankernel);
        deviceHelper->device.freeDevice(gaussiankernel_d);
    }
};

}
