#include "VshipAPI.h"
#include<iostream>
#include<string>
#include<vector>
#include<fstream>
#include<stdlib.h>
#include<chrono>
#include<thread>
#include "../src/ffvship_utility/CLI_Parser.hpp"

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetErrorMessage(err, errmsg, 1024);\
    printf("Vship Error occured: %s", errmsg);\
    exit(1);\
}

enum class MetricType { SSIMULACRA2, Butteraugli, CVVDP};

class MetricComputeClass{
private:
    Vship_Colorspace_t colorspace;
    const uint8_t* csrc1[3];
    const uint8_t* csrc2[3];
public:
    MetricType metric;
    Vship_SSIMU2Handler ssimu2Handle;
    Vship_ButteraugliHandler butterHandle;
    Vship_CVVDPHandler cvvdpHandle;
    MetricComputeClass(MetricType metric, Vship_Colorspace_t colorspace, const uint8_t* csrc1[3], const uint8_t* csrc2[3]) : metric(metric), colorspace(colorspace) {
        this->csrc1[0] = csrc1[0];
        this->csrc1[1] = csrc1[1];
        this->csrc1[2] = csrc1[2];
        this->csrc2[0] = csrc2[0];
        this->csrc2[1] = csrc2[1];
        this->csrc2[2] = csrc2[2];
        switch (metric){
            case MetricType::SSIMULACRA2:
                ErrorCheck(Vship_SSIMU2Init(&ssimu2Handle, colorspace, colorspace));
                break;
            case MetricType::Butteraugli:
                ErrorCheck(Vship_ButteraugliInit(&butterHandle, colorspace, colorspace, 2, 80));
                break;
            case MetricType::CVVDP:
                ErrorCheck(Vship_CVVDPInit(&cvvdpHandle, colorspace, colorspace, 0, false, "standard_fhd"));
                break;
        }
    }
    //true if the result is correctly perfect, false otherwise
    bool compute(){
        const int64_t lineSize[3] = {colorspace.width, colorspace.width, colorspace.width};
        bool out = false;
        double score = 0;
        Vship_ButteraugliScore butterScore = {0, 0, 0};
        switch (metric){
            case MetricType::SSIMULACRA2:
                ErrorCheck(Vship_ComputeSSIMU2(ssimu2Handle, &score, csrc1, csrc2, lineSize, lineSize));
                if (score > 99) out = true;
                break;
            case MetricType::Butteraugli:
                ErrorCheck(Vship_ComputeButteraugli(butterHandle, &butterScore, NULL, 0, csrc1, csrc2, lineSize, lineSize));
                if (butterScore.norminf == 0) out = true;
                break;
            case MetricType::CVVDP:
                ErrorCheck(Vship_ComputeCVVDP(cvvdpHandle, &score, NULL, 0, csrc1, csrc2, lineSize, lineSize));
                if (score == 10) out = true;
                break;
        }
        return out;
    }
    ~MetricComputeClass(){
        switch (metric){
            case MetricType::SSIMULACRA2:
                ErrorCheck(Vship_SSIMU2Free(ssimu2Handle));
                break;
            case MetricType::Butteraugli:
                ErrorCheck(Vship_ButteraugliFree(butterHandle));
                break;
            case MetricType::CVVDP:
                ErrorCheck(Vship_CVVDPFree(cvvdpHandle));
                break;
        }
    }
};

//seconds
void computeNtimes(double* output, int n, MetricType metric, Vship_Colorspace_t colorspace, const uint8_t* csrc1[3], const uint8_t* csrc2[3]){
    MetricComputeClass computeClass(metric, colorspace, csrc1, csrc2);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++){
        computeClass.compute();
    }
    auto end = std::chrono::high_resolution_clock::now();
    *output = std::chrono::duration<double>(end-start).count();
}

MetricType parse_metric_name(const std::string &name) {
    std::string lowered;
    lowered.resize(name.size());
    for (unsigned int i = 0; i < name.size(); i++){
        lowered[i] = std::tolower(name[i]);
    }
    if (lowered == "ssimulacra2" || lowered == "ssimu2") return MetricType::SSIMULACRA2;
    if (lowered == "butteraugli" || lowered == "butter") return MetricType::Butteraugli;
    if (lowered == "cvvdp") return MetricType::CVVDP;
    return MetricType::SSIMULACRA2;
}

int main(int argc, char* argv[]){
    std::cerr << "---speed-test-libvship---" << std::endl;
    std::vector<std::string> args(argc);
    for (int i = 0; i < argc; i++){
        args[i] = argv[i];
    }
    helper::ArgParser parser;

    std::string metric_name = "ssimu2";
    parser.add_flag({"--metric", "-m"}, &metric_name, "Which metric to use [SSIMULACRA2, Butteraugli, CVVDP]");
    MetricType metric = parse_metric_name(metric_name);

    int width = 1920;
    int height = 1080;
    parser.add_flag({"--width"}, &width, "width of image to speedtest");
    parser.add_flag({"--height"}, &height, "height of image to speedtest");

    int gpu_threads = 1;
    parser.add_flag({"-g"}, &gpu_threads, "number of concurrent gpu threads");
    int nCompute = 100;
    parser.add_flag({"-n"}, &nCompute, "number of frame to compute");

    int gpu_id = 0;
    parser.add_flag({"--gpu-id"}, &gpu_id, "gpu id chosen to run test");

    if (parser.parse_cli_args(args) != 0) { //the parser will have already printed an error
        return 1;
    }

    Vship_Version vshipVer = Vship_GetVersion();
    std::cout << "Running Vship Version " << vshipVer.major << "." << vshipVer.minor << "." << vshipVer.minorMinor << " Under Backend ";
    if (vshipVer.backend == Vship_Cuda) {
        std::cout << "Cuda";
    } else if (vshipVer.backend == Vship_Vulkan) {
        std::cout << "Vulkan";
    } else if (vshipVer.backend == Vship_HIP) {
        std::cout << "HIP";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;

    int numgpu;
    ErrorCheck(Vship_GetDeviceCount(&numgpu));

    Vship_DeviceInfo devinfo;
    for (int i = 0; i < numgpu; i++){
        ErrorCheck(Vship_GetDeviceInfo(&devinfo, i));
        std::cout << ((gpu_id == i) ? "> [" : "- ") << i << ": " << devinfo.name << ((gpu_id == i) ? "]" : "") << std::endl;
    }

    ErrorCheck(Vship_GPUFullCheck(gpu_id));
    ErrorCheck(Vship_SetDevice(gpu_id));

    //init images, they will not be modified and will be reused by mulitple metric threads
    uint8_t* src1[3];
    uint8_t* src2[3];
    src1[0] = (uint8_t*)malloc(sizeof(uint8_t)*width*height*6);
    src1[1] = src1[0]+sizeof(uint8_t)*width*height;
    src1[2] = src1[1]+sizeof(uint8_t)*width*height;
    src2[0] = src1[2]+sizeof(uint8_t)*width*height;
    src2[1] = src2[0]+sizeof(uint8_t)*width*height;
    src2[2] = src2[1]+sizeof(uint8_t)*width*height;

    for (uint64_t i = 0; i < width*height*6; i++){
        src1[0][i] = 0;
    }

    const uint8_t* csrc1[3] = {src1[0], src1[1], src1[2]};
    const uint8_t* csrc2[3] = {src2[0], src2[1], src2[2]};

    Vship_Colorspace_t colorspace;
    colorspace.width = width;
    colorspace.height = height;
    colorspace.target_width = -1;
    colorspace.target_height = -1;
    colorspace.sample = Vship_SampleUINT8;
    colorspace.range = Vship_RangeFull;
    colorspace.subsampling = {0, 0};
    colorspace.chromaLocation = Vship_ChromaLoc_Left;
    colorspace.YUVMatrix = Vship_MATRIX_RGB;
    colorspace.transferFunction = Vship_TRC_Linear;
    colorspace.primaries = Vship_PRIMARIES_BT709;
    colorspace.crop = {0, 0, 0, 0};

    std::vector<std::thread> threads;
    std::vector<double> results(gpu_threads);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < gpu_threads; i++){
        threads.emplace_back(computeNtimes, &results[i], nCompute, metric, colorspace, csrc1, csrc2);
    }
    for (int i = 0; i < gpu_threads; i++){
        threads[i].join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(end-start).count();

    std::cout << "individual times: ";
    for (int i = 0; i < gpu_threads; i++){
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "total time taken by metrics and thread launching: " << total << " s (" << gpu_threads*nCompute/total << " fps)" << std::endl;

    free(src1[0]);

    return 0;
}
