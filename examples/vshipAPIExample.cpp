#include "VshipAPI.h"
#include <iostream>
#include <stdlib.h>
#include <cstring>

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetDetailedLastError(errmsg, 1024);\
    std::cout << "Vship Error occured: " << errmsg;\
    exit(1);\
}

//YUV420 uint8 format
class Image{
private:
    int small_width;
    int small_height;
    uint8_t* src[3];
public:
    int image_width;
    int image_height;
    const uint8_t* csrc[3];
    int64_t lineSize[3];
    Image(const int image_width, const int image_height) : image_width(image_width), image_height(image_height){
        small_width = (image_width+1)/2;
        small_height = (image_height+1)/2;
        src[0] = (uint8_t*)malloc(image_width*image_height);
        src[1] = (uint8_t*)malloc(small_width*small_height);
        src[2] = (uint8_t*)malloc(small_width*small_height);
        for (int i = 0; i < 3; i++) csrc[i] = src[i];
        lineSize[0] = image_width;
        lineSize[1] = small_width;
        lineSize[2] = small_width;
    }
    void set(const uint8_t val){
        memset(src[0], val, image_width*image_height);
        memset(src[1], val, small_width*small_height);
        memset(src[2], val, small_width*small_height);
    }
    void setPix(const uint8_t val, const int plane, const int x, const int y){
        int stride = (plane == 0) ? image_width : small_width;
        src[plane][y*stride+x] = val;
    }
    ~Image(){
        free(src[0]);
        free(src[1]);
        free(src[2]);
    }
};

void* getButteraugliHandler(Vship_Colorspace_t src, Vship_Colorspace_t dst, float intensity_multiplier, int Qnorm){
    Vship_InitButteraugli_1 initStruct = {
        Vship_StructInitButteraugli_1,
        src, dst,
        Qnorm,
        intensity_multiplier,
        0 //gpu_id
    };
    void* handler;
    ErrorCheck(Vship_InitHandler(&handler, (void*)&initStruct));
    return handler;
}

int main(){
    ErrorCheck(Vship_GPUFullCheck(0));

    const int image_width = 1920;
    const int image_height = 1080;

    //we supposed the same for source and distorted but they can have different colorspaces
    //here we define the classic Limited BT709 YUV420 source
    Vship_Colorspace_t colorspace = {0};
    colorspace.width = image_width;
    colorspace.height = image_height;
    colorspace.target_width = -1;
    colorspace.target_height = -1;
    colorspace.sample = Vship_SampleUINT8;
    colorspace.range = Vship_RangeLimited;
    colorspace.subsampling = {1, 1};
    colorspace.colorFamily = Vship_ColorYUV;
    colorspace.YUVMatrix = Vship_MATRIX_BT709;
    colorspace.transferFunction = Vship_TRC_BT709;
    colorspace.primaries = Vship_PRIMARIES_BT709;

    void* butterhandler = getButteraugliHandler(colorspace, colorspace, 203.f, 5);

    Image im1(image_width, image_height);
    Image im2(image_width, image_height);
    im1.set(128u);
    im2.set(128u);
    //create a difference close to the middle in chroma
    im1.setPix(150u, 2, image_width/4-2, image_height/4-2);

    //the result will be here!
    //let's allocate the space required to store the distortion map. It is always float
    //you can use image_width as a stride in this case
    float* distortionMap = (float*)malloc(sizeof(float)*image_height*image_width);

    Vship_ScoreButteraugli scoreStruct = {
        Vship_StructScoreButteraugli,
        0, 0, 0, //output norms
        (uint8_t*)distortionMap, //distmap pointer
        image_width*sizeof(float) //distmap stride
    };

    //ask vship to compute!
    //it will handle color conversion and metric computing
    ErrorCheck(Vship_ComputeHandler(butterhandler, &scoreStruct, im1.csrc, im2.csrc, im1.lineSize, im2.lineSize));

    //now scoreStruct contains butteraugli scores and distortionMap contains the distortion map!
    std::cout << "Butteraugli norms (3/5/inf): " << scoreStruct.norm3 << "/" << scoreStruct.normQ << "/" << scoreStruct.norminf << std::endl;

    //error in middle of image:
    std::cout << "Error in middle of image: " << distortionMap[image_width*image_height/2 + image_width/2] << std::endl;

    //free the distortion map we just allocated
    free(distortionMap);

    ErrorCheck(Vship_FreeHandler(butterhandler));

    return 0;
}
