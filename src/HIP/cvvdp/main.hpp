#pragma once

#include "../util/preprocessor.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../../util/concurrency.hpp"
#include "../gpuColorToLinear/vshipColor.hpp"

#include "parameters.hpp"
#include "display_models.hpp"
#include "colors.hpp"
#include "resize.hpp"
#include "temporalFilter.hpp"
#include "pooling.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include "gaussianBlur.hpp"
#include "maskingModel.hpp"
#include "distmap_specifics.hpp"

namespace cvvdp{

double CVVDPprocess(const uint8_t *dstp, int64_t dststride, int64_t resize_width, int64_t resize_height, TemporalRing& temporalRing1, TemporalRing& temporalRing2, CSF_Handler& csfhandle, GaussianHandle& gaussianhandle, DisplayModel* model, hipStream_t stream1, hipStream_t stream2, hipEvent_t event1, hipEvent_t event2){
    const int64_t minwidth = temporalRing1.width;
    const int64_t minheight = temporalRing1.height;
    const int64_t width = resize_width;
    const int64_t height = resize_height;

    //if minwidth < resize_width it means we need to resize still, this requires more memory

    int allocatedPlanes = 9;
    int64_t resizeBufferSize = 0;
    if (minwidth < resize_width) resizeBufferSize = resize_width*minheight*sizeof(float);

    const int64_t pyramidSize = LpyrMemoryNeedPerPlane(width, height, model->get_screen_ppd());
    //std::cout << "allocation bytes : " << sizeof(float)*(allocatedPlanes+stream1_supPlane)*pyramidSize << std::endl;
    float* mem_d;
    hipError_t erralloc = hipMallocAsync(&mem_d, sizeof(float)*(allocatedPlanes)*pyramidSize+resizeBufferSize, stream1);
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }
    //second stream
    float* mem_d2;
    erralloc = hipMallocAsync(&mem_d2, sizeof(float)*allocatedPlanes*pyramidSize+resizeBufferSize, stream2);
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }

    //memory process is that we first put inputs on [mem_d, mem_d+4*pyramidSize[
    //then we have a big merged kernel that fills in [mem_d+4*pyramidSize, mem_d+5*pyramidSize[ (Lbkg), that fills next scale in the input buffer and fills the output in [mem_d+5*pyramidSize, mem_d+9*pyramidSize[
    //then we have another big kernel with all the gaussian blurs and up to final map that reads from [mem_d+4*pyramidSize, mem_d+9*pyramidSize[ and outputs to [mem_d, mem_d+4*pyramidSize[

    const std::vector<float> band_frequencies = get_frequencies(width, height, model->get_screen_ppd());
    const int levels = band_frequencies.size();
    int64_t w = width;
    int64_t h = height;
    std::vector<int64_t> bandOffsets(levels);
    bandOffsets[0] = 0;
    for (int i = 1; i < levels; i++){
        bandOffsets[i] = bandOffsets[i-1]+w*h;
        w = (w+1)/2;
        h = (h+1)/2;
    }

    std::vector<std::vector<float*>> memPlace1;
    std::vector<std::vector<float*>> memPlace2;
    memPlace1.resize(9);
    memPlace2.resize(9);
    for (int channel = 0; channel < memPlace1.size(); channel++){
        memPlace1[channel].resize(levels);
        memPlace2[channel].resize(levels);
        for (int band = 0; band < levels; band++){
            memPlace1[channel][band] = mem_d+channel*pyramidSize+bandOffsets[band];
            memPlace2[channel][band] = mem_d2+channel*pyramidSize+bandOffsets[band];
        }
    }

    //let's get the temporal channel out of the temporal ring!
    computeTemporalChannels(temporalRing1, memPlace1[0][0], memPlace1[1][0], memPlace1[2][0], memPlace1[3][0], stream1);
    computeTemporalChannels(temporalRing2, memPlace2[0][0], memPlace2[1][0], memPlace2[2][0], memPlace2[3][0], stream2);

    if (minwidth < resize_width){
        //this means that the planes contained in Y_sustained... are not the right size!
        //we need to resize now
        resizePlane(memPlace1[0][0], memPlace1[4][0], memPlace1[0][0], minwidth, minheight, width, height, stream1);
        resizePlane(memPlace1[1][0], memPlace1[4][0], memPlace1[1][0], minwidth, minheight, width, height, stream1);
        resizePlane(memPlace1[2][0], memPlace1[4][0], memPlace1[2][0], minwidth, minheight, width, height, stream1);
        resizePlane(memPlace1[3][0], memPlace1[4][0], memPlace1[3][0], minwidth, minheight, width, height, stream1);
        resizePlane(memPlace2[0][0], memPlace2[4][0], memPlace2[0][0], minwidth, minheight, width, height, stream2);
        resizePlane(memPlace2[1][0], memPlace2[4][0], memPlace2[1][0], minwidth, minheight, width, height, stream2);
        resizePlane(memPlace2[2][0], memPlace2[4][0], memPlace2[2][0], minwidth, minheight, width, height, stream2);
        resizePlane(memPlace2[3][0], memPlace2[4][0], memPlace2[3][0], minwidth, minheight, width, height, stream2);
    }

    //stream1 waits for stream2 to be done
    GPU_CHECK(hipEventRecord(event1, stream2));
    GPU_CHECK(hipStreamWaitEvent(stream1, event1));
    //now we only use stream1, this is the time for merging both sources

    const float ppd = model->get_screen_ppd();

    w = width;
    h = height;
    for (int band = 0; band < levels; band++){
        if (band != levels-1){
            for (int channel = 0; channel < 4; channel++){
                fromPyrReduceToCSF(memPlace1[0][band], memPlace2[0][band], pyramidSize, w, h, band, channel, csfhandle, stream1);
                //preGaussianPreCompute(memPlace1[4][band], memPlace1[5+channel][band], memPlace2[5+channel][band], w, h, channel, band, csfhandle, stream1);
            }
            computeD(memPlace1[0][band], memPlace1[1][band], memPlace1[2][band], memPlace1[3][band], memPlace1[5][band], memPlace1[6][band], memPlace1[7][band], memPlace1[8][band], memPlace2[5][band], memPlace2[6][band], memPlace2[7][band], memPlace2[8][band], w, h, gaussianhandle, stream1);
        } else {
            //here Lbkg is different, it is the mean.
            //now we need to take the mean.
            computeMean<1>(memPlace1[0][band], memPlace1[4][band], w*h, true, stream1);
            computeMean<1>(memPlace2[0][band], memPlace2[4][band], w*h, true, stream1);
            for (int channel = 0; channel < 4; channel++){
                baseBandPyrRefine<true, 1>(memPlace1[channel+5][band], memPlace1[channel][band], memPlace1[4][band], w*h, stream1);
                baseBandPyrRefine<true, 1>(memPlace2[channel+5][band], memPlace2[channel][band], memPlace2[4][band], w*h, stream1);
                computeD_baseband(memPlace1[channel][band], memPlace1[4][band], memPlace1[5+channel][band], memPlace2[5+channel][band], w, h, channel, band, csfhandle, stream1);
            }
        }

        w = (w+1)/2;
        h = (h+1)/2;
    }
    
    //stream2 waits for stream1 to be done before freeing memory (memory of Lpyr2)
    GPU_CHECK(hipEventRecord(event2, stream1));
    GPU_CHECK(hipStreamWaitEvent(stream2, event2));
    GPU_CHECK(hipFreeAsync(mem_d2, stream2));
    //the complete full distortion map is stored in Lpyr1
    //and we can use Lpyr1 Lbkg place as temporary storage for pooling
    std::vector<float> scores(levels*4); //per_band and per channel
    int tempOffset = 0;
    float* temp = memPlace1[4][0];
    w = width;
    h = height;
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const int64_t size = w*h;
            computeMean<2>(memPlace1[channel][band], temp+tempOffset, size, true, stream1);
            tempOffset++; //we stored the norm at temp[tempOffset]
        }
        w = (w+1)/2;
        h = (h+1)/2;
    }
    GPU_CHECK(hipMemcpyDtoHAsync(scores.data(), temp, sizeof(float)*levels*4, stream1));

    if (dstp != NULL){
        //we want a distmap
        float* res[4] = {memPlace1[0][0], memPlace1[1][0], memPlace1[2][0], memPlace1[3][0]};
        getDistMap(res, bandOffsets, width, height, levels, stream1);
        strideAdder(res[0], res[1], dststride, width, height, stream1);
        GPU_CHECK(hipMemcpyDtoHAsync((void*)(dstp), res[1], dststride * height, stream1));
    }

    GPU_CHECK(hipStreamSynchronize(stream1));
    GPU_CHECK(hipFreeAsync(mem_d, stream1));

    //we have our levels*4 scores
    double finalValue = 0;
    for (int band = 0; band < levels; band++){
        for (int channel = 0; channel < 4; channel++){
            const float val = scores[band*4+channel];
            //std::cout << "val: " << val << " at channel " << channel << " band " << band << std::endl;
            finalValue += std::pow(val, 4);
        }
    }
    finalValue = std::pow(finalValue, 1./4.);

    return finalValue;

    //GPU_CHECK(hipFreeAsync(mem_d, stream1));
    //GPU_CHECK(hipFreeAsync(mem_d2, stream2));
    //GPU_CHECK(hipStreamSynchronize(stream1));
    //GPU_CHECK(hipStreamSynchronize(stream2));
    //return 0.f;
}

//currently at 32 planes + 3*fps/2 planes consumed
class CVVDPComputingImplementation{
    DisplayModel* model = NULL;
    float fps = 0;
    //std::vector<float> all_scores; //for past frames
    int numFrame = 0;
    double score_squareSum = 0; //to avoid recomputing it all the time
    TemporalRing temporalRing1; //source
    TemporalRing temporalRing2; //encoded
    CSF_Handler csf_handler;
    GaussianHandle gaussianhandle;
    VshipColorConvert::Converter converter1;
    VshipColorConvert::Converter converter2;
    Vship_Colorspace_t ref_colorspace;
    Vship_Colorspace_t dis_colorspace;
    int64_t source_width = 0;
    int64_t source_height = 0;
    hipStream_t stream1 = 0;
    hipStream_t stream2 = 0;
    hipEvent_t event1;
    hipEvent_t event2;
    int gpu_id;
public:
    int64_t resize_width = 0;
    int64_t resize_height = 0;
    void init(Vship_Colorspace_t source_colorspace, Vship_Colorspace_t source_colorspace2, int gpu_id, float fps, bool resizeToDisplay, std::string model_key, std::string model_config_json){
        this->gpu_id = gpu_id;
        GPU_CHECK(hipSetDevice(gpu_id));
        model = new DisplayModel(model_key, model_config_json);

        converter1.init(source_colorspace, VshipColorConvert::linRGB, stream1);
        converter2.init(source_colorspace2, VshipColorConvert::linRGB, stream2);

        source_width = converter1.getWidth();
        source_height = converter1.getHeight();

        //assert they have the same width/height
        if (converter2.getWidth() != source_width || converter2.getHeight() != source_height){
            throw VshipError(DifferingInputType, __FILE__, __LINE__);            
        }

        if (resizeToDisplay){
            //we resize to match width since display_width/width <= display_height/height
            if (source_width*model->resolution[1] >= source_height*model->resolution[0]){
                resize_width = model->resolution[0];
                resize_height = (source_height*model->resolution[0])/source_width;
            } else {
                //here we resize to match source_height
                resize_height = model->resolution[1];
                resize_width = (source_width*model->resolution[1])/source_height;
            }
        } else {
            resize_width = source_width;
            resize_height = source_height;
        }

        //std::cout << "base/resize : " << width << "x" << source_height << "/" << resize_width << "x" << resize_height << std::endl;

        this->fps = fps;
        ref_colorspace = source_colorspace;
        dis_colorspace = source_colorspace2;

        const uint levels = get_frequencies(resize_width, resize_height, model->get_screen_ppd()).size();

        GPU_CHECK(hipEventCreate(&event1));
        GPU_CHECK(hipEventCreate(&event2));
        GPU_CHECK(hipStreamCreate(&stream1));
        GPU_CHECK(hipStreamCreate(&stream2));

        //temporalRing are big VRAM consumers, we ll make themm store the smallest version of the video
        temporalRing1.init(fps, std::min(resize_width, source_width), std::min(resize_height, source_height));
        temporalRing2.init(fps, std::min(resize_width, source_width), std::min(resize_height, source_height));
        csf_handler.init(resize_width, resize_height, model->get_screen_ppd());
        gaussianhandle.init();
        score_squareSum = 0;
        numFrame = 0;
    }
    void destroy(){
        GPU_CHECK(hipSetDevice(gpu_id));
        temporalRing1.destroy();
        temporalRing2.destroy();
        csf_handler.destroy();
        gaussianhandle.destroy();
        converter1.destroy();
        converter2.destroy();
        delete model;
        GPU_CHECK(hipStreamDestroy(stream1));
        GPU_CHECK(hipStreamDestroy(stream2));
        GPU_CHECK(hipEventDestroy(event1));
        GPU_CHECK(hipEventDestroy(event2));
    }
    void loadImageToRing(const uint8_t *srcp1[3], const uint8_t *srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        GPU_CHECK(hipSetDevice(gpu_id));
        bool is_resized = (source_width != resize_width || source_height != resize_height);
        int64_t bufferSize = 0;
        if (is_resized){
            if (source_width < resize_width){
                bufferSize = 0;
                //we will put source size into the temporal ring -> no resize here
            } else {
                //we need to resize before putting in temporal ring.
                bufferSize += source_width*source_height*3*sizeof(float)+source_height*resize_width*sizeof(float);
            }
        } else {
            //we put everything in the temporalRing directly
            bufferSize = 0;
        }
        //allocate memory to send the raw to gpu
        float* mem_d = NULL;
        float* mem_d2 = NULL;
        if (bufferSize > 0){
            hipError_t erralloc = hipMallocAsync(&mem_d, (bufferSize), stream1);
            if (erralloc != hipSuccess){
                throw VshipError(OutOfVRAM, __FILE__, __LINE__);
            }
            //for second stream
            erralloc = hipMallocAsync(&mem_d2, (bufferSize), stream2);
            if (erralloc != hipSuccess){
                throw VshipError(OutOfVRAM, __FILE__, __LINE__);
            }
        }

        //defined onnly if we resize in this function
        float* tempResize = mem_d + source_width*source_height*3; //of size resize temp
        //for second stream
        float* tempResize2 = mem_d2 + source_width*source_height*3; //of size resize temp

        //free up a plane by increasing history to 1, we can now edit the frame 0 which is blank
        temporalRing1.rotate();
        temporalRing2.rotate();
        //take color planes from the ring
        float* source_ptr = temporalRing1.getFramePointer(0);
        float* encoded_ptr = temporalRing2.getFramePointer(0);

        //final destination
        int64_t minwidth = std::min(source_width, resize_width);
        int64_t minheight = std::min(source_height, resize_height);

        float* src1_d[3] = {source_ptr, source_ptr+minwidth*minheight, source_ptr+2*minwidth*minheight};
        float* src2_d[3] = {encoded_ptr, encoded_ptr+minwidth*minheight, encoded_ptr+2*minwidth*minheight};

        float* base_plane1[3];
        float* base_plane2[3];
        //we work on the temporalRing buffer if source fits (=> smaller than resize)
        if (is_resized && source_width > resize_width){
            base_plane1[0] = mem_d;
            base_plane1[1] = base_plane1[0] + source_width*source_height;
            base_plane1[2] = base_plane1[0] + 2*source_width*source_height;

            base_plane2[0] = mem_d2;
            base_plane2[1] = base_plane2[0] + source_width*source_height;
            base_plane2[2] = base_plane2[0] + 2*source_width*source_height;
        } else {
            for (int i = 0; i < 3; i++){
                base_plane1[i] = src1_d[i];
                base_plane2[i] = src2_d[i];
            }
        }

        converter1.convert(base_plane1, srcp1, lineSize);
        converter2.convert(base_plane2, srcp2, lineSize2);

        //resize as early as possible if source bigger than resize
        if (is_resized && source_width > resize_width){
            for (int i = 0; i < 3; i++){
                resizePlane(src1_d[i], tempResize, base_plane1[i], source_width, source_height, resize_width, resize_height, stream1);
                resizePlane(src2_d[i], tempResize2, base_plane2[i], source_width, source_height, resize_width, resize_height, stream2);
            }
        }
        //everything is in temporalRing buffer so we can free memory already
        if (bufferSize > 0){
            GPU_CHECK(hipFreeAsync(mem_d, stream1));
            GPU_CHECK(hipFreeAsync(mem_d2, stream2));
        }

        //now we work on src_d which has size minimum of source and resize

        const float Y_peak = model->max_luminance;
        const float Y_black = model->getBlackLevel();
        const float Y_refl = model->getReflLevel();
        const float exposure = model->exposure;
        const DisplayColorspace isHDR = model->colorspace;

        //std::cout << "Y_peak, Y_black, Y_refl, exposure : " << Y_peak << " " <<  Y_black << " " << Y_refl << " " << exposure << std::endl;

        //we put the frame's planes on GPU

        //colorspace conversion at this point
        linRGB_to_dkl_DisplayEncode(src1_d, ref_colorspace.primaries, ref_colorspace.transferFunction, Y_peak, Y_black, Y_refl, exposure, isHDR, minwidth*minheight, stream1);
        linRGB_to_dkl_DisplayEncode(src2_d, dis_colorspace.primaries, dis_colorspace.transferFunction, Y_peak, Y_black, Y_refl, exposure, isHDR, minwidth*minheight, stream2);
        //and we are done, the frame is loaded in the ring with the right colorspace, next step is temporal filtering
    }
    double run(const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
        loadImageToRing(srcp1, srcp2, lineSize, lineSize2);
        const double current_score = CVVDPprocess(dstp, dststride, resize_width, resize_height, temporalRing1, temporalRing2, csf_handler, gaussianhandle, model, stream1, stream2, event1, event2);
        score_squareSum += std::pow(current_score, beta_t);
        double resQ;
        numFrame++;
        if (numFrame == 1){
            resQ = current_score * image_int;
        } else {
            resQ = std::pow(score_squareSum/(double)numFrame, 1./beta_t);
        }
        return toJOD(resQ);
    }
    void flushOnlyScore(){
        numFrame = 0;
        score_squareSum = 0.;
    }
    //empties the history.
    void flushTemporalRing(){
        numFrame = 0;
        score_squareSum = 0.;
        temporalRing1.reset();
        temporalRing2.reset();
    }
};

}
