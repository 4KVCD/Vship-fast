#pragma once

extern "C" {
#include <ffms.h>
#include <libavutil/pixfmt.h>
}

#include "CLI_Parser.hpp"
#include "ffmpegToVshipColorFormat.hpp"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <cstring>
#include <bit>
#include <cmath>

#ifndef ASSERT_WITH_MESSAGE
#define ASSERT_WITH_MESSAGE(condition, message)\
if (!(condition)) {\
    std::fprintf(stderr, "Assertion failed!\nExpression : %s\nFile       : %s\n  Line       : %d\nMessage    : %s\n", #condition, __FILE__, __LINE__, message);\
    std::abort();\
}
#endif

enum class MetricType { SSIMULACRA2, Butteraugli, CVVDP, Unknown };

//only works on powers of 2 bytes, T must be unsigned
template<typename T>
constexpr static T reverseBits(T a){
    T mask = (1llu << sizeof(T)*8) - (1llu << (sizeof(T)*4));
    uint size = sizeof(T)*4;
    while (size >= 1){
        a = ((a & mask) >> size) | ((a & (~mask)) << size);
        size >>= 1;
        mask = (mask)^(mask >> (size));
    }
    return a;
}

//only works on powers of 2 bytes, T must be unsigned
template<typename T>
constexpr static T reverseBytes(T a){
    T mask = (1llu << sizeof(T)*8) - (1llu << (sizeof(T)*4));
    uint size = sizeof(T)*4;
    while (size >= 8){
        a = ((a & mask) >> size) | ((a & (~mask)) << size);
        size >>= 1;
        mask = (mask)^(mask >> (size));
    }
    return a;
}

struct MetricParameters{
    //general
    int gpu_id = 0;

    //CVVDP
    bool resizeToDisplay = 0;
    std::string model_key = "standard_fhd";
    std::string model_config_json = "";

    //butteraugli
    int intensity_target_nits = 203;
    int Qnorm = 2;
};

class GpuWorker {
  private:
    Vship_Colorspace_t image_colorspace;
    int64_t lineSize[3];

    Vship_Colorspace_t encoded_colorspace;
    int64_t lineSize2[3];

    MetricType selected_metric;
    Vship_Handle handler;

  public:
    GpuWorker(MetricType metric, Vship_Colorspace_t source_colorspace, Vship_Colorspace_t encoded_colorspace, const int64_t lineSize[3], const int64_t lineSize2[3], float fps, MetricParameters metricParam){
        selected_metric = metric;
        this->image_colorspace = source_colorspace;
        this->encoded_colorspace = encoded_colorspace;
        for (int i = 0; i < 3; i++){
            this->lineSize[i] = lineSize[i];
            this->lineSize2[i] = lineSize2[i];
        }
        allocate_gpu_memory(fps, metricParam);
    }
    ~GpuWorker(){
        deallocate_gpu_memory();
    }

    std::pair<std::tuple<double, double, double>, Vship_Exception>
    compute_metric_score(const uint8_t* srcp1[3], const uint8_t* srcp2[3]) {

        if (selected_metric == MetricType::SSIMULACRA2) {
            Vship_ScoreSSIMULACRA2 scoreStruct;
            scoreStruct.structType = Vship_StructScoreSSIMULACRA2;
            Vship_Exception err = Vship_ComputeHandler(handler, (void*)&scoreStruct, srcp1, srcp2, lineSize, lineSize2);
            if (err != Vship_NoError){
                char errmsg[1024];
                int l = Vship_GetDetailedLastErrorHandler(handler, errmsg, 1024);
                if (l == 0) {
                    //error occured in the return error
                    Vship_GetDetailedLastError(errmsg, 1024);
                }
                std::cerr << " error: " << errmsg << std::endl;
            }
            double s = scoreStruct.score;
            return {{s, s, s}, err};
        }

        if (selected_metric == MetricType::Butteraugli) {
            Vship_ScoreButteraugli butterscore;
            butterscore.structType = Vship_StructScoreButteraugli;
            butterscore.dstp = nullptr;
            butterscore.dststride = 0;
            Vship_Exception err = Vship_ComputeHandler(handler, (void*)&butterscore, srcp1, srcp2, lineSize, lineSize2);
            if (err != Vship_NoError){
                char errmsg[1024];
                int l = Vship_GetDetailedLastErrorHandler(handler, errmsg, 1024);
                if (l == 0) {
                    //error occured in the return error
                    Vship_GetDetailedLastError(errmsg, 1024);
                }
                std::cerr << " error: " << errmsg << std::endl;
            }
            return {{butterscore.normQ, butterscore.norm3, butterscore.norminf}, err};
        }

        if (selected_metric == MetricType::CVVDP){
            Vship_ScoreCVVDP cvvdpScore;
            cvvdpScore.structType = Vship_StructScoreCVVDP;
            cvvdpScore.dstp = nullptr;
            cvvdpScore.dststride = 0;
            Vship_Exception err = Vship_ComputeHandler(handler, (void*)&cvvdpScore, srcp1, srcp2, lineSize, lineSize2);
            if (err != Vship_NoError){
                char errmsg[1024];
                int l = Vship_GetDetailedLastErrorHandler(handler, errmsg, 1024);
                if (l == 0) {
                    //error occured in the return error
                    Vship_GetDetailedLastError(errmsg, 1024);
                }
                std::cerr << " error: " << errmsg << std::endl;
            }
            double s = cvvdpScore.score;
            return {{s, s, s}, err};
        }

        ASSERT_WITH_MESSAGE(false, "Unknown metric specified for GpuWorker.");
        return {{0.0f, 0.0f, 0.0f}, Vship_BadErrorType};
    }

    static uint8_t *allocate_external_rgb_buffer(uint64_t bytes, int gpu_id) {
        const size_t buffer_size_bytes = bytes * 3;
        uint8_t *buffer_ptr = nullptr;

        const Vship_Exception result = Vship_PinnedMalloc2(
            reinterpret_cast<void **>(&buffer_ptr), buffer_size_bytes, gpu_id);

        ASSERT_WITH_MESSAGE(
            result == Vship_NoError && buffer_ptr != nullptr,
            "Pinned buffer allocation failed in allocate_external_rgb_buffer");

        return buffer_ptr;
    }

    static void deallocate_external_buffer(uint8_t *buffer_ptr, int gpu_id) {
        if (buffer_ptr != nullptr) {
            Vship_PinnedFree2(buffer_ptr, gpu_id);
        }
    }

  private:
    void allocate_gpu_memory(float fps, MetricParameters metricParam) {
        Vship_Exception err;
        if (selected_metric == MetricType::SSIMULACRA2) {
            Vship_InitSSIMULACRA2_1 initStruct;
            initStruct.structType = Vship_StructInitSSIMULACRA2_1;
            initStruct.src_colorspace = image_colorspace;
            initStruct.dis_colorspace = encoded_colorspace;
            initStruct.gpu_id = metricParam.gpu_id;
            err = Vship_InitHandler(&handler, (void*)&initStruct);
        } else if (selected_metric == MetricType::Butteraugli) {
            Vship_InitButteraugli_1 initStruct;
            initStruct.structType = Vship_StructInitButteraugli_1;
            initStruct.src_colorspace = image_colorspace;
            initStruct.dis_colorspace = encoded_colorspace;
            initStruct.Qnorm = metricParam.Qnorm;
            initStruct.intensity_multiplier = metricParam.intensity_target_nits;
            initStruct.gpu_id = metricParam.gpu_id;
            err = Vship_InitHandler(&handler, (void*)&initStruct);
        } else if (selected_metric == MetricType::CVVDP){
            Vship_InitCVVDP_1 initStruct;
            initStruct.structType = Vship_StructInitCVVDP_1;
            initStruct.src_colorspace = image_colorspace;
            initStruct.dis_colorspace = encoded_colorspace;
            initStruct.fps = fps;
            initStruct.resizeToDisplay = metricParam.resizeToDisplay;
            initStruct.model_key_cstr = metricParam.model_key.c_str();
            initStruct.model_config_json_cstr = metricParam.model_config_json.c_str();
            initStruct.gpu_id = metricParam.gpu_id;
            err = Vship_InitHandler(&handler, (void*)&initStruct);
        } else {
            ASSERT_WITH_MESSAGE(false,
                                "Unknown metric during memory allocation.");
        }
        if (err != Vship_NoError){
            char errmsg[1024];
            Vship_GetDetailedLastError(errmsg, 1024);
            std::cerr << errmsg << std::endl;
            ASSERT_WITH_MESSAGE(false, "Failed to initialize GPU Worker");
            return;
        }
    }

    void deallocate_gpu_memory() {
        Vship_FreeHandler(handler);
    }
};

int IndexCallback(int64_t current, int64_t total, void* progp){
    auto prog = (ProgressBar<100, false>*)progp;
    prog->set_values(current/1000000, total/1000000);
    return 0;
}

class FFMSIndexResult {
  private:
    static constexpr int error_message_buffer_size = 1024;

    char error_message_buffer[error_message_buffer_size]{};
    FFMS_ErrorInfo error_info;

  public:
    std::string file_path;
    std::string index_file_path;
    FFMS_Index* index = NULL;
    FFMS_Track* track = NULL;
    int selected_video_track = -1;
    int numFrame = 0;
    int write = 0;

    explicit FFMSIndexResult(const std::string& input_file_path, std::string input_index_file_path, const bool cache_index, const bool debug_out = false, const bool show_progress = true) {
        file_path = input_file_path;
        index_file_path = input_index_file_path;
        FFMS_Init(0, 0);

        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = error_message_buffer_size;
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;

        bool from_file_success = false;

        //if cached but no path specified, we default to this path
        if (input_index_file_path == "" && cache_index) input_index_file_path = input_file_path + ".ffindex";

        //if path not empty, we try to read
        if (input_index_file_path != ""){
            index = FFMS_ReadIndex(input_index_file_path.c_str(), &error_info);
            if (index != nullptr && !FFMS_IndexBelongsToFile(index, input_file_path.c_str(), &error_info)) {
                from_file_success = true;
                if (debug_out) std::cout << "Successfully read index from [" << input_index_file_path << "]" << std::endl;
            } else {
                if (debug_out) std::cout << "Index file at [" << input_index_file_path << "] is invalid or does not exist, creating" << std::endl;
            }
        }

        //if failed, we will need to compute ourself
        if (!from_file_success){
            FFMS_Indexer *indexer = FFMS_CreateIndexer(input_file_path.c_str(), &error_info);
            ASSERT_WITH_MESSAGE(indexer != nullptr,
                            ("FFMS2: Failed to create indexer for file [" +
                            input_file_path + "] - " + error_message_buffer)
                                .c_str());

            ProgressBar<100, false>* prog;
            if (show_progress){
                std::cout << "Indexing Progress (MB) " << input_file_path << std::endl;
                prog = new ProgressBar<100, false>(0);
                FFMS_SetProgressCallback(indexer, IndexCallback, prog);
            }

            index = FFMS_DoIndexing2(indexer, FFMS_IEH_ABORT, &error_info);
            ASSERT_WITH_MESSAGE(index != nullptr,
                            ("FFMS2: Failed to index file [" + input_file_path +
                            "] - " + error_message_buffer)
                                .c_str());

            if (show_progress){
                prog->refresh(true);
                delete prog;
                std::cout << std::endl; //end of progressbar
            }
        }

        //if we need to cache and it s computed (compiler will optimize automatically)
        //we will write the cache file
        if (!from_file_success && cache_index){
            write = FFMS_WriteIndex(input_index_file_path.c_str(), index, &error_info);
            ASSERT_WITH_MESSAGE(write == 0,
                                ("FFMS2: Failed to write index to file [" + input_index_file_path +
                                "] - " + error_message_buffer)
                                    .c_str());
            if (debug_out) std::cout << "Successfully wrote index to [" << input_index_file_path << "]" << std::endl;
        }

        selected_video_track =
            FFMS_GetFirstTrackOfType(index, FFMS_TYPE_VIDEO, &error_info);
        ASSERT_WITH_MESSAGE(selected_video_track >= 0,
                            ("FFMS2: No video track found in file [" +
                             input_file_path + "] - " + error_message_buffer)
                                .c_str());

        track =
            FFMS_GetTrackFromIndex(index, selected_video_track);
        ASSERT_WITH_MESSAGE(track != NULL,
                            ("FFMS2: Failed to get FFMS_Track in file [" +
                             input_file_path + "]").c_str());

        numFrame = FFMS_GetNumFrames(track);
        ASSERT_WITH_MESSAGE(numFrame != 0, ("FFMS2: Got 0 frames in [" +
                             input_file_path + "]").c_str());

    }

    std::set<int> getKeyFrameIndices(){
        std::set<int> out;
        for (int i = 0; i < numFrame; i++){
            const FFMS_FrameInfo* frameinfo = FFMS_GetFrameInfo(track, i);
            ASSERT_WITH_MESSAGE(frameinfo != NULL, ("Failed to retrieve KeyFrame information in the indexer for file [" + file_path + "]").c_str());
            if (frameinfo->KeyFrame != 0){
                out.insert(i);
            }
        }
        return out;
    }

    ~FFMSIndexResult() {
        if (index != nullptr) {
            FFMS_DestroyIndex(index);
            index = nullptr;
        }

        selected_video_track = -1;
    }
};

class FFMSFrameReader {
  private:
    int num_decoder_threads = std::thread::hardware_concurrency();
    int seek_mode = FFMS_SEEK_NORMAL;

  public:
    const FFMS_VideoProperties *video_properties = nullptr;
    const FFMS_Frame *current_frame = nullptr;

    AVPixelFormat video_pixel_format = AV_PIX_FMT_NONE;
    int frame_width = 0;
    int frame_height = 0;
    int total_frame_count = 0;
    float fps = 0;

    FFMS_VideoSource *video_source = nullptr;

    FFMS_ErrorInfo error_info;
    char error_message_buffer[1024] = {};

    explicit FFMSFrameReader(const std::string &file_path, FFMS_Index *index,
                             int video_track_index) {
        initialize_error_info();
        create_video_source(file_path, index, video_track_index);
        load_video_properties();
        load_first_frame();
        configure_pixel_format(file_path);
    }

    ~FFMSFrameReader() {
        if (video_source != nullptr) {
            FFMS_DestroyVideoSource(video_source);
            video_source = nullptr;
        }
    }

    void fetch_frame(int frame_index) {
        current_frame = FFMS_GetFrame(video_source, frame_index, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch frame [" +
                             std::to_string(frame_index) + "] - " +
                             error_message_buffer)
                                .c_str());
    }

  private:
    void initialize_error_info() {
        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = sizeof(error_message_buffer);
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;
    }

    void create_video_source(const std::string &file_path, FFMS_Index *index,
                             int track_index) {

        if (file_path.substr(file_path.size()-4) == ".jpg" || file_path.substr(file_path.size()-5) == ".jpeg"){
            //std::cerr << "jpg detected, seek mode workaround for ffms2 enabled" << std::endl;
            seek_mode = FFMS_SEEK_LINEAR;
        }

        video_source =
            FFMS_CreateVideoSource(file_path.c_str(), track_index, index,
                                   num_decoder_threads, seek_mode, &error_info);

        ASSERT_WITH_MESSAGE(video_source != nullptr,
                            ("FFMS2: Failed to create video source for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }

    void load_video_properties() {
        video_properties = FFMS_GetVideoProperties(video_source);
        ASSERT_WITH_MESSAGE(video_properties != nullptr,
                            "FFMS2: FFMS_GetVideoProperties returned null.");

        total_frame_count = video_properties->NumFrames;
        ASSERT_WITH_MESSAGE(total_frame_count > 0,
                            "FFMS2: No frames found in video stream.");

        fps = (float)video_properties->FPSNumerator / (float)video_properties->FPSDenominator;
    }

    void load_first_frame() {
        current_frame = FFMS_GetFrame(video_source, 0, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch first frame - " +
                             std::string(error_message_buffer))
                                .c_str());

        video_pixel_format =
            static_cast<AVPixelFormat>(current_frame->EncodedPixelFormat);
        frame_width = current_frame->EncodedWidth;
        frame_height = current_frame->EncodedHeight;
    }

    void configure_pixel_format(const std::string &file_path) {
        int output_formats[] = {static_cast<int>(video_pixel_format), -1};

        int result = FFMS_SetOutputFormatV2(video_source, output_formats,
                                            frame_width, frame_height,
                                            FFMS_RESIZER_BICUBIC, &error_info);

        ASSERT_WITH_MESSAGE(result == 0,
                            ("FFMS2: Failed to set output format for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }
};

class BasicConverterProcessor {
  public:
    AVPixelFormat src_pixfmt;
    int64_t width;
    int64_t height;
    int64_t unpack_stride[3];
    int64_t planeSizeUnpack[3];
    bool require_unpack = true;

    BasicConverterProcessor(const FFMS_Frame *ref_frame) {
        initialize_formats(ref_frame);
        build_unpack(ref_frame);
    }

    ~BasicConverterProcessor() {
    }

    //dst planes should have size given by planeSizeUnpack
    void process(const FFMS_Frame *src, uint8_t *dst[3]) {
        unpack(src, dst);
    }

  private:
    void initialize_formats(const FFMS_Frame *frame) {

        width = frame->EncodedWidth;
        height = frame->EncodedHeight;
        src_pixfmt = (AVPixelFormat)frame->EncodedPixelFormat;
    }

    void build_unpack(const FFMS_Frame *frame){
        //list supported formats (they are repeated in ffmpegToZimgFormat to handle them correctly)
        require_unpack = true;
        switch (src_pixfmt){

            //bitdepth 8, 422
            case AV_PIX_FMT_UYVY422:
            case AV_PIX_FMT_YUYV422:
            unpack_stride[0] = width;
            unpack_stride[1] = (width/2);
            unpack_stride[2] = unpack_stride[1]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            /*
            //bitdepth 16, 422
            case AV_PIX_FMT_Y216LE:
            unpack_stride[0] = width*2;
            unpack_stride[1] = (width);
            unpack_stride[2] = unpack_stride[1]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;
            */

            //bitdepth 8, 444
            case AV_PIX_FMT_BGR24:
            case AV_PIX_FMT_RGB24:
            case AV_PIX_FMT_ARGB:
            case AV_PIX_FMT_RGBA:
            case AV_PIX_FMT_ABGR:
            case AV_PIX_FMT_BGRA:
            case AV_PIX_FMT_GBRP:
            unpack_stride[0] = width;
            unpack_stride[1] = unpack_stride[0];
            unpack_stride[2] = unpack_stride[0]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            //depth 16, 444
            case AV_PIX_FMT_RGB48BE:
            case AV_PIX_FMT_RGBA64BE:
            case AV_PIX_FMT_RGB48LE:
            case AV_PIX_FMT_RGBA64LE:
            unpack_stride[0] = width*2;
            unpack_stride[1] = unpack_stride[0];
            unpack_stride[2] = unpack_stride[0]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            //depth 32, 444
            case AV_PIX_FMT_RGBF32LE:
            case AV_PIX_FMT_GBRPF32LE:
            unpack_stride[0] = width*4;
            unpack_stride[1] = unpack_stride[0];
            unpack_stride[2] = unpack_stride[0]; //same for both chroma

            planeSizeUnpack[0] = unpack_stride[0]*height;
            planeSizeUnpack[1] = unpack_stride[1]*height;
            planeSizeUnpack[2] = unpack_stride[2]*height;
            break;

            default:
                Vship_Colorspace_t colorspace;
                ffmpegToVshipFormat(colorspace, frame);
                unpack_stride[0] = frame->Linesize[0];
                unpack_stride[1] = frame->Linesize[1];
                unpack_stride[2] = frame->Linesize[2];
                planeSizeUnpack[0] = unpack_stride[0]*height;
                planeSizeUnpack[1] = (height >> colorspace.subsampling.subh)*unpack_stride[1];
                planeSizeUnpack[2] = (height >> colorspace.subsampling.subh)*unpack_stride[2];
                require_unpack = false;
                return; //no unpack
        }
    }

    void unpack(const FFMS_Frame *src, uint8_t* unpack_buffer[3]){
        if (!require_unpack){
            memcpy(unpack_buffer[0], src->Data[0], planeSizeUnpack[0]);
            memcpy(unpack_buffer[1], src->Data[1], planeSizeUnpack[1]);
            memcpy(unpack_buffer[2], src->Data[2], planeSizeUnpack[2]);
            return;
        }
        switch (src_pixfmt){
            /*
            case AV_PIX_FMT_Y216LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width/2; i++){
                        //Y
                        ((uint16_t*)(unpack_buffer[0]))[j*unpack_stride[0]+2*i] = ((uint16_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        //U
                        ((uint16_t*)(unpack_buffer[1]))[j*unpack_stride[1]+i] = ((uint16_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        //Y
                        ((uint16_t*)(unpack_buffer[0]))[j*unpack_stride[0]+2*i+1] = ((uint16_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        //V
                        ((uint16_t*)(unpack_buffer[2]))[j*unpack_stride[2]+i] = ((uint16_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            */
            case AV_PIX_FMT_YUYV422:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width/2; i++){
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        //U
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i+1] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        //V
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_UYVY422:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width/2; i++){
                        //U
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        //V
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i+1] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_RGBF32LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((float*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((float*)(src->Data[0]+j*src->Linesize[0]))[3*i];
                        ((float*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((float*)(src->Data[0]+j*src->Linesize[0]))[3*i+1];
                        ((float*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((float*)(src->Data[0]+j*src->Linesize[0]))[3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_GBRPF32LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((float*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((float*)(src->Data[2]+j*src->Linesize[2]))[i];
                        ((float*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((float*)(src->Data[0]+j*src->Linesize[0]))[i];
                        ((float*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((float*)(src->Data[1]+j*src->Linesize[1]))[i];
                    }
                }
            break;
            case AV_PIX_FMT_RGB48BE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i]);
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+1]);
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+2]);
                    }
                }
            break;
            case AV_PIX_FMT_RGBA64BE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i]);
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+1]);
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = reverseBytes(((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+2]);
                    }
                }
            break;
            case AV_PIX_FMT_RGB48LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA64LE:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_BGR24:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+1];
                        unpack_buffer[2][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGB24:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_ARGB:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_ABGR:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                    }
                }
            break;
            case AV_PIX_FMT_BGRA:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+0];
                    }
                }
            break;
            case AV_PIX_FMT_GBRP:
                for (int j = 0; j < height; j++){
                    for (int i = 0; i < width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[2]))[j*src->Linesize[0]+i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[1]+i];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[1]))[j*src->Linesize[2]+i];
                    }
                }
            break;

            default:
                //this should not happen and result of forgiveness of the dev, better to place this in case
                std::cout << "Error: Trying to unpack an unsupported format " << src_pixfmt << " line " << __LINE__ << " of " << __FILE__ << std::endl;
                return;
        }
    }
};

class VideoManager {
  public:
    Vship_Colorspace_t colorspace;
    std::unique_ptr<FFMSFrameReader> reader;
    std::unique_ptr<BasicConverterProcessor> processor;

    VideoManager(const std::string &file_path, FFMS_Index *index, int video_track_index) {

        reader = std::make_unique<FFMSFrameReader>(file_path, index,
                                                   video_track_index);

        processor = std::make_unique<BasicConverterProcessor>(reader->current_frame);

        int ret = ffmpegToVshipFormat(colorspace, reader->current_frame);
        //define it entirely to avoid undefined values wandering around
        colorspace.target_width = -1;
        colorspace.target_height = -1;
        colorspace.crop = {0, 0, 0, 0};

        ASSERT_WITH_MESSAGE(
            processor != nullptr,
            "VideoManager: Failed to initialize ZimgProcessor.");

        ASSERT_WITH_MESSAGE(
            ret == 0,
            "Video Format Not Supported.");
    }
    int64_t getBufferSize(){
        return processor->planeSizeUnpack[0]+processor->planeSizeUnpack[1]+processor->planeSizeUnpack[2];
    }
    void fetch_frame_into_buffer(int frame_index, uint8_t *output_buffer) {
        reader->fetch_frame(frame_index);
        uint8_t* srcp[3] = {output_buffer, output_buffer+processor->planeSizeUnpack[0], output_buffer+processor->planeSizeUnpack[0]+processor->planeSizeUnpack[1]};
        processor->process(reader->current_frame, srcp);
    }
};

void printColorspace(Vship_Colorspace_t colorspace){
    std::cout << "Source Size: " << colorspace.width << "x" << colorspace.height << std::endl;
    std::cout << "Resize To: " << colorspace.target_width << "x" << colorspace.target_height << std::endl;
    std::cout << "Cropped by (Top/Bottom/Left/Right): " << colorspace.crop.top << "/" << colorspace.crop.bottom << "/" << colorspace.crop.left << "/" << colorspace.crop.right << std::endl;
    std::cout << "=> Converted Size: ";
    if (colorspace.target_width != -1){
        std::cout << colorspace.target_width-colorspace.crop.right-colorspace.crop.left;
    } else {
        std::cout << colorspace.width-colorspace.crop.right-colorspace.crop.left;
    }
    std::cout << "x";
    if (colorspace.target_height != -1){
        std::cout << colorspace.target_height-colorspace.crop.top-colorspace.crop.bottom;
    } else {
        std::cout << colorspace.height-colorspace.crop.top-colorspace.crop.bottom;
    }
    std::cout << std::endl;
    std::cout << "Sample Type: ";
    switch(colorspace.sample){
        case Vship_SampleUINT8:
            std::cout << "Uint8_t";
            break;
        case Vship_SampleUINT9:
            std::cout << "Uint9_t";
            break;
        case Vship_SampleUINT10:
            std::cout << "Uint10_t";
            break;
        case Vship_SampleUINT12:
            std::cout << "Uint12_t";
            break;
        case Vship_SampleUINT14:
            std::cout << "Uint14_t";
            break;
        case Vship_SampleUINT16:
            std::cout << "Uint16_t";
            break;
        case Vship_SampleHALF:
            std::cout << "Half";
            break;
        case Vship_SampleFLOAT:
            std::cout << "Float";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Color Family: ";
    if (colorspace.colorFamily == Vship_ColorRGB){
        std::cout << "RGB";
    } else if (colorspace.colorFamily == Vship_ColorYUV){
        std::cout << "YUV";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Range: ";
    if (colorspace.range == Vship_RangeLimited){
        std::cout << "Limited";
    } else if (colorspace.range == Vship_RangeFull){
        std::cout << "Full";
    } else {
        std::cout << "Unknown";
    }
    std::cout << std::endl;
    std::cout << "Subsampling (log): " << colorspace.subsampling.subw << "x" << colorspace.subsampling.subh << std::endl;
    std::cout << "Chroma Location: ";
    switch(colorspace.chromaLocation){
        case Vship_ChromaLoc_Center:
            std::cout << "Center";
            break;
        case Vship_ChromaLoc_TopLeft:
            std::cout << "Top-Left";
            break;
        case Vship_ChromaLoc_Left:
            std::cout << "Left";
            break;
        case Vship_ChromaLoc_Top:
            std::cout << "Top";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "YUV Matrix: ";
    switch(colorspace.YUVMatrix){
        case Vship_MATRIX_RGB:
            std::cout << "RGB";
            break;
        case Vship_MATRIX_BT709:
            std::cout << "BT709";
            break;
        case Vship_MATRIX_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_MATRIX_ST170_M:
            std::cout << "ST170_M";
            break;
        case Vship_MATRIX_BT2020_NCL:
            std::cout << "BT2020_NCL";
            break;
        case Vship_MATRIX_BT2020_CL:
            std::cout << "BT2020_CL";
            break;
        case Vship_MATRIX_BT2100_ICTCP:
            std::cout << "BT2100_ICTCP";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Transfer Function: ";
    switch(colorspace.transferFunction){
        case Vship_TRC_BT709:
            std::cout << "BT709";
            break;
        case Vship_TRC_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_TRC_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_TRC_BT601:
            std::cout << "BT601";
            break;
        case Vship_TRC_Linear:
            std::cout << "Linear";
            break;
        case Vship_TRC_sRGB:
            std::cout << "sRGB";
            break;
        case Vship_TRC_PQ:
            std::cout << "PQ";
            break;
        case Vship_TRC_ST428:
            std::cout << "ST428";
            break;
        case Vship_TRC_HLG:
            std::cout << "HLG";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;

    std::cout << "Primaries: ";
    switch(colorspace.primaries){
        case Vship_PRIMARIES_INTERNAL:
            std::cout << "XYZ";
            break;
        case Vship_PRIMARIES_BT709:
            std::cout << "BT709";
            break;
        case Vship_PRIMARIES_BT470_M:
            std::cout << "BT470_M";
            break;
        case Vship_PRIMARIES_BT470_BG:
            std::cout << "BT470_BG";
            break;
        case Vship_PRIMARIES_BT2020:
            std::cout << "BT2020";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << std::endl;
}

void printVulkanFeatureMatrix(Vship_VulkanFeatureMatrix& features){
    std::cout << "storageBuffer16BitAccess : " << features.storageBuffer16BitAccess << std::endl;
    std::cout << "uniformAndStorageBuffer16BitAccess : " << features.uniformAndStorageBuffer16BitAccess << " (not required)" << std::endl;
    std::cout << "variablePointersStorageBuffer : " << features.variablePointersStorageBuffer << std::endl;
    std::cout << "variablePointers : " << features.variablePointers << std::endl;
    std::cout << "storagePushConstant8 : " << features.storagePushConstant8 << " (not required)" << std::endl;
    std::cout << "storageBuffer8BitAccess : " << features.storageBuffer8BitAccess << std::endl;
    std::cout << "uniformAndStorageBuffer8BitAccess : " << features.uniformAndStorageBuffer8BitAccess << " (not required)" << std::endl;
    std::cout << "shaderFloat16 : " << features.shaderFloat16 << " (not required in default build)" << std::endl;
    std::cout << "shaderFloat64 : " << features.shaderFloat64 << " (not required)" << std::endl;
    std::cout << "shaderInt8 : " << features.shaderInt8 << std::endl;
    std::cout << "shaderInt16 : " << features.shaderInt16 << std::endl;
    std::cout << "shaderInt64 : " << features.shaderInt64 << std::endl;
    std::cout << "bufferDeviceAddress : " << features.bufferDeviceAddress << std::endl;
    std::cout << "shaderStorageBufferArrayDynamicIndexing : " << features.shaderStorageBufferArrayDynamicIndexing << std::endl;
    std::cout << "synchronization2 : " << features.synchronization2 << std::endl;
    std::cout << "logicOp : " << features.logicOp << " (not required)" << std::endl;
}

struct CommandLineOptions {
    std::string source_file;
    std::string encoded_file;
    std::string json_output_file;
    std::string svg_output_file;
    std::string source_index;
    std::string encoded_index;

    int start_frame = 0;
    int end_frame = -1;
    int every_nth_frame = 1;
    int encoded_offset = 0;

    Vship_CropRectangle_t cropSource = {0, 0, 0, 0};
    Vship_CropRectangle_t cropEncoded = {0, 0, 0, 0};

    std::vector<int> source_indices_list;
    std::vector<int> encoded_indices_list;

    MetricParameters metricParam;

    int gpu_threads = 3;
    int cpu_threads = 1;

    bool version = false;
    bool list_gpus = false;
    bool gpu_info = false;
    bool verbose = false;
    MetricType metric = MetricType::SSIMULACRA2; //SSIMULACRA2 by default

    bool NoAssertExit = false; //please exit without creating an assertion failed scary error

    bool live_index_score_output = false;

    bool cache_index = false;
};

std::vector<int> splitPerToken(std::string inp){
    std::vector<int> out;
    std::string temp;

    for (const char c: inp){
        switch (c){
            case ',':
                out.push_back(std::stoi(temp));
                temp.clear();
                break;
            case ' ':
                continue;
            default:
                temp.push_back(c);
        }
    }
    if (!temp.empty()) out.push_back(std::stoi(temp));

    return out;
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
    return MetricType::Unknown;
}

CommandLineOptions parse_command_line_arguments(int argc, char **argv) {
    std::vector<std::string> args(argc);
    for (int i = 0; i < argc; i++){
        args[i] = argv[i];
    }

    helper::ArgParser parser;

    std::string metric_name;
    std::string source_indices_str;
    std::string encoded_indices_str;

    CommandLineOptions opts;

    parser.add_flag({"--source", "-s"}, &opts.source_file, "Reference video to compare to", true);
    parser.add_flag({"--encoded", "-e"}, &opts.encoded_file, "Distorted encode of the source", true);
    parser.add_flag({"--metric", "-m"}, &metric_name, "Which metric to use [SSIMULACRA2, Butteraugli, CVVDP]");
    parser.add_flag({"--json"}, &opts.json_output_file, "Outputs metric results to a json file");
    parser.add_flag({"--svg"}, &opts.svg_output_file, "Outputs metric results to a svg file as a graph");
    parser.add_flag({"--live-score-output"}, &opts.live_index_score_output, "replace stdout output with index-score lines");
    parser.add_flag({"--source-index"}, &opts.source_index, "FFMS2 index file for source video");
    parser.add_flag({"--encoded-index"}, &opts.encoded_index, "FFMS2 index file for encoded video");
    parser.add_flag({"--cache-index"}, &opts.cache_index, "Write index files to disk and reuse if available");

    parser.add_flag({"--start"}, &opts.start_frame, "Starting frame of source");
    parser.add_flag({"--end"}, &opts.end_frame, "Ending frame of source");
    parser.add_flag({"--encoded-offset"}, &opts.encoded_offset, "Frame offset of encoded video to source");
    parser.add_flag({"--every"}, &opts.every_nth_frame, "Frame sampling rate");
    parser.add_flag({"--source-indices"}, &source_indices_str, "List of source indices subjective to --start, --end, --every and --encoded-offset. If --encoded-indices isnt specified, this will be applied to encoded-indices too. Format is integers separated by comma");
    parser.add_flag({"--encoded-indices"}, &encoded_indices_str, "List of encoded indices subjective to --start, --end, --every and --encoded-offset. Format is integers separated by comma");

    parser.add_flag({"--cropTopSource"}, &opts.cropSource.top, "Allows to crop source");
    parser.add_flag({"--cropBottomSource"}, &opts.cropSource.bottom, "Allows to crop source");
    parser.add_flag({"--cropLeftSource"}, &opts.cropSource.left, "Allows to crop source");
    parser.add_flag({"--cropRightSource"}, &opts.cropSource.right, "Allows to crop source");
    parser.add_flag({"--cropTopEncoded"}, &opts.cropEncoded.top, "Allows to crop Encoded");
    parser.add_flag({"--cropBottomEncoded"}, &opts.cropEncoded.bottom, "Allows to crop Encoded");
    parser.add_flag({"--cropLeftEncoded"}, &opts.cropEncoded.left, "Allows to crop Encoded");
    parser.add_flag({"--cropRightEncoded"}, &opts.cropEncoded.right, "Allows to crop Encoded");

    parser.add_flag({"--resizeToDisplay"}, &opts.metricParam.resizeToDisplay, "Allow to resize to the screen resolution specified in the model (default off)");
    parser.add_flag({"--displayModel"}, &opts.metricParam.model_key, "Allow specifying screen disposition to CVVDP (default standard_fhd)");
    parser.add_flag({"--displayConfig"}, &opts.metricParam.model_config_json, "Allow specifying an external json display configuration path");
    parser.add_flag({"--intensity-target"}, &opts.metricParam.intensity_target_nits, "Target nits for Butteraugli");
    parser.add_flag({"--qnorm"}, &opts.metricParam.Qnorm, "Optional Norm to compute (default to 2)");
    parser.add_flag({"--threads", "-t"}, &opts.cpu_threads, "Number of Decoder process, recommended is 2");
    parser.add_flag({"--gpu-threads", "-g"}, &opts.gpu_threads, "GPU thread count, recommended is 3");
    parser.add_flag({"--gpu-id"}, &opts.metricParam.gpu_id, "GPU index");
    parser.add_flag({"--list-gpu"}, &opts.list_gpus, "List available GPUs");
    parser.add_flag({"--gpu-info"}, &opts.gpu_info, "Gives Extended Informations on the selected GPU");
    parser.add_flag({"--version"}, &opts.version, "Print FFVship version");
    parser.add_flag({"--verbose"}, &opts.verbose, "Print Colorspace found in Source and encoded");

    if (parser.parse_cli_args(args) != 0) { //the parser will have already printed an error
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.list_gpus || opts.version || opts.gpu_info) return opts;

    try {
        opts.source_indices_list = splitPerToken(source_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --source-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }
    try {
        opts.encoded_indices_list = splitPerToken(encoded_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --encoded-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.source_file.empty()){
        std::cerr << "Source file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (opts.encoded_file.empty()){
        std::cerr << "Encoded file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (!metric_name.empty()) {
        opts.metric = parse_metric_name(metric_name);
        if (opts.metric == MetricType::Unknown){
            std::cerr << "Unknown metric type. Expected 'SSIMULACRA2', 'Butteraugli' or 'CVVDP'." << std::endl;
            opts.NoAssertExit = true;
        }
    }

    return opts;
}

const double jod_a = 0.0439569391310215;
const double jod_exp = 0.9302042722702026;
const double jod_linear_slope = jod_a * std::pow(0.1, jod_exp-1.); //toJOD turns linear below 0.1
const double jod_linear_limit = 10. - jod_linear_slope * 0.1;
const double cvvdp_image_int = 0.577918291091919;

double JOD(double a){
    if (a > 0.1) return 10. - jod_a * std::pow(a, jod_exp);
    return 10. - jod_linear_slope * a;
}
double inverse_JOD(double a){
    if (a < jod_linear_limit) return std::pow((10.-a)/jod_a, 1./jod_exp);
    return (10.-a)/jod_linear_slope;
}

struct svg_bucket_t {
    double lo, hi, avg;
};

struct svg_serie_t {
    std::vector<svg_bucket_t> column;
    std::string name;
    double mean, lo, hi;
};

std::string svg_escaped_name(const std::string& path){
    const size_t pos = path.find_last_of("/\\");
    std::string out;
    for (const char c: path.substr((pos == std::string::npos) ? 0 : pos+1)){
        switch (c){
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default: out += c;
        }
    }
    return out;
}

std::string svg_num(double value, int significant = 4){
    char out[32];
    std::snprintf(out, sizeof(out), "%.*g", significant, value);
    return out;
}

//1, 2, 2.5 or 5 times a power of ten, so that gridlines land on readable values
double svg_nice_step(double span, int target, bool integer_only){
    if (!(span > 0.) || target < 1) return 1.;
    const double magnitude = std::pow(10., std::floor(std::log10(span/(double)target)));
    const double ladder[5] = {1., 2., 2.5, 5., 10.};
    double best = 1., best_error = 1e300;
    for (int i = 0; i < 5; i++){
        const double step = ladder[i]*magnitude;
        if (integer_only && (step < 1. || step != std::floor(step))) continue;
        const double error = std::fabs(span/step - (double)target);
        if (error < best_error){ best_error = error; best = step; }
    }
    return best;
}

//every column holds the lowest, highest and mean score of the frames it covers
svg_serie_t svg_make_serie(const std::vector<double>& scores, int columns, int stride, int shift, bool cvvdp){
    const int64_t frames = (int64_t)(scores.size()/stride);
    svg_serie_t serie;
    double all_lo = 1e300, all_hi = -1e300, all_sum = 0., cumul = 0.;
    int64_t frame = 0;

    serie.column.resize(columns);
    for (int c = 0; c < columns; c++){
        const int64_t column_end = (int64_t)(c+1)*frames/columns;
        const int64_t count = column_end - frame;
        const double column_cumul = cumul;
        double lo = 1e300, hi = -1e300, sum = 0.;

        for (; frame < column_end; frame++){
            double value;
            if (cvvdp){
                //cvvdp reports running pool, not frame score: Q(i) = (sum(q^2)/(i+1))^0.5 (q(0) scaled by image_int)
                //undoing needs the cumulated sum in double, float loses the frame difference on long clips
                const double pooled = inverse_JOD(scores[frame]);
                const double next = (frame == 0) ? (pooled/cvvdp_image_int)*(pooled/cvvdp_image_int) : pooled*pooled*(double)(frame+1);
                value = JOD(std::sqrt(std::max(0., next - cumul)));
                cumul = next;
            } else {
                value = scores[frame*stride + shift]; //butteraugli holds its 3 norms interleaved
            }
            if (!std::isfinite(value)) continue;
            lo = std::min(lo, value);
            hi = std::max(hi, value);
            sum += value;
        }

        if (lo > hi){ //column held nothing usable
            const double previous = (c > 0) ? serie.column[c-1].avg : 0.;
            serie.column[c] = {previous, previous, previous};
            continue;
        }
        //cumulated sum telescopes over the column & cancels the reconstruct noise
        //that single frame differences carry. cvvdp only shows the pooled column value
        double avg = sum/(double)count;
        if (cvvdp){
            avg = JOD(std::sqrt(std::max(0., (cumul-column_cumul)/(double)count)));
            lo = hi = avg;
        }
        serie.column[c] = {lo, hi, avg};
        all_lo = std::min(all_lo, lo);
        all_hi = std::max(all_hi, hi);
        all_sum += sum;
    }

    if (all_lo > all_hi) all_lo = all_hi = 0.;
    serie.lo = all_lo;
    serie.hi = all_hi;
    serie.mean = all_sum/(double)frames;
    return serie;
}

int write_to_svg(const CommandLineOptions& cli_args, const std::vector<double>& source_scores) {
    if (cli_args.metric == MetricType::Unknown) return 1;

    const bool is_butter = (cli_args.metric == MetricType::Butteraugli);
    const bool is_cvvdp = (cli_args.metric == MetricType::CVVDP);
    const char* metric_name = is_butter ? "Butteraugli" : (is_cvvdp ? "CVVDP" : "SSIMULACRA2");
    const int nbr_serie = is_butter ? 3 : 1;
    const int64_t nbr_frames = (int64_t)(source_scores.size()/nbr_serie);

    if (nbr_frames <= 0){
        std::cerr << "No score to write to the svg output file" << std::endl;
        return 1;
    }

    const double width = 1920., height = 1080.;
    const double margin_left = 124., margin_right = 44., margin_top = 104., margin_bottom = 96.;
    const double plot_width = width - margin_left - margin_right;
    const double plot_height = height - margin_top - margin_bottom;
    const double plot_right = margin_left + plot_width;
    const double plot_bottom = margin_top + plot_height;

    //one column per horizontal pixel: a million frames still gives a small, renderable svg
    const int nbr_column = (int)std::min(nbr_frames, (int64_t)plot_width);
    const char* colors[3] = {"#00A3EE", "#8DA600", "#FF4593"};

    std::vector<svg_serie_t> series(nbr_serie);
    for (int s = 0; s < nbr_serie; s++) series[s] = svg_make_serie(source_scores, nbr_column, nbr_serie, s, is_cvvdp);
    if (is_butter){
        series[0].name = std::to_string(cli_args.metricParam.Qnorm)+"-Norm";
        series[1].name = "3-Norm";
        series[2].name = "INF-Norm";
    } else {
        series[0].name = metric_name;
        //cvvdp's own pooled result is the last reported value, a mean of reconstructed frames would be noise
        if (is_cvvdp) series[0].mean = source_scores[nbr_frames-1];
    }

    double min_value = 1e300, max_value = -1e300;
    for (const svg_serie_t& serie: series){
        min_value = std::min(min_value, serie.lo);
        max_value = std::max(max_value, serie.hi);
    }
    if (max_value - min_value < 1e-9*std::max(1., std::fabs(max_value))){ //flat, or a single frame
        const double spread = std::max(0.5, std::fabs(max_value)*0.01);
        min_value -= spread;
        max_value += spread;
    }
    const double margin_value = std::min(0.3, (max_value-min_value)*0.05);
    min_value = std::max(min_value - margin_value, is_butter ? 0. : -1e300);
    max_value = std::min(max_value + margin_value, is_butter ? 1e300 : (is_cvvdp ? 10. : 100.));
    if (!(max_value > min_value)) min_value = max_value - 1.;

    const double value_span = max_value - min_value;
    const double value_step = svg_nice_step(value_span, 20, false);
    const double frame_span = (double)std::max((int64_t)1, nbr_frames-1);

    const bool custom_indices = !cli_args.source_indices_list.empty() || !cli_args.encoded_indices_list.empty();
    const int64_t frame_step = custom_indices ? 1 : std::max(1, cli_args.every_nth_frame);
    const int64_t label_first = custom_indices ? 0 : std::max(0, cli_args.start_frame);
    const int64_t label_last = label_first + (nbr_frames-1)*frame_step;
    const int64_t label_step = (int64_t)(svg_nice_step((double)(label_last-label_first), 16, true) + 0.5);

    std::vector<double> value_tick;
    for (int64_t i = (int64_t)std::ceil(min_value/value_step); i <= (int64_t)std::floor(max_value/value_step); i++){
        const double value = (double)i*value_step;
        value_tick.push_back((std::fabs(value) < value_step*1e-6) ? 0. : value);
    }
    std::vector<int64_t> frame_tick;
    for (int64_t label = ((label_first+label_step-1)/label_step)*label_step; label <= label_last; label += label_step){
        frame_tick.push_back(label);
    }

    std::ofstream svgfile(cli_args.svg_output_file, std::ios_base::out);
    if (!svgfile) {
        std::cerr << "Failed to open svg output file" << std::endl;
        return 1;
    }
    svgfile << std::fixed << std::setprecision(1);

    auto xpos = [&](double frame){ return margin_left + plot_width*frame/frame_span; };
    auto ypos = [&](double value){
        if (!std::isfinite(value)) value = min_value;
        return plot_bottom - plot_height*std::min(1., std::max(0., (value-min_value)/value_span));
    };
    auto tick_x = [&](int64_t label){ return xpos((double)(label-label_first)/(double)frame_step); };
    auto group = [&](const char* attributes){ svgfile << "<g " << attributes << ">\n"; };
    auto close_group = [&](){ svgfile << "</g>\n"; };
    auto text = [&](double x, double y, const std::string& content, double dy = 0.){
        svgfile << "<text x=\"" << x << "\" y=\"" << y << "\" dy=\"" << dy << "em\">" << content << "</text>\n";
    };
    auto line = [&](double x1, double y1, double x2, double y2){
        svgfile << "<line x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y2 << "\" />\n";
    };
    auto rect = [&](double x, double y, double w, double h, const char* attributes){
        svgfile << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << w << "\" height=\"" << h << "\" " << attributes << " />\n";
    };

    std::string subtitle = svg_escaped_name(cli_args.source_file) + " vs " + svg_escaped_name(cli_args.encoded_file)
                         + " &#183; " + std::to_string(nbr_frames) + (nbr_frames > 1 ? " frames" : " frame");
    if (nbr_column < nbr_frames){
        subtitle += " &#183; " + std::to_string(nbr_frames/nbr_column) + " frames per column ("
                  + (is_cvvdp ? "pooled" : "min/max band + mean") + ")";
    }

    svgfile << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svgfile << "<svg width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height
            << "\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n";
    rect(0., 0., width, height, "fill=\"#000000\"");
    group("font-family=\"DejaVu Sans, Verdana, sans-serif\" text-anchor=\"middle\"");

    group("fill=\"#FFFFFF\" font-size=\"34px\"");
    text(width*0.5, 46., metric_name);
    close_group();

    group("fill=\"#9A9A9A\" font-size=\"19px\"");
    text(width*0.5, 78., subtitle);
    close_group();

    group("stroke=\"#808080\" stroke-width=\"1.5\" stroke-opacity=\"0.28\"");
    for (const double value: value_tick) line(margin_left, ypos(value), plot_right, ypos(value));
    for (const int64_t label: frame_tick) line(tick_x(label), margin_top, tick_x(label), plot_bottom);
    close_group();

    group("fill=\"#C8C8C8\" font-size=\"17px\" text-anchor=\"end\"");
    for (const double value: value_tick) text(margin_left-14., ypos(value), svg_num(value, 6), 0.35);
    close_group();

    group("fill=\"#C8C8C8\" font-size=\"17px\"");
    for (const int64_t label: frame_tick) text(tick_x(label), plot_bottom+16., std::to_string(label), 0.7);
    close_group();

    std::vector<double> xs(nbr_column);
    for (int64_t c = 0, frame = 0; c < nbr_column; c++){
        const int64_t column_end = (c+1)*nbr_frames/nbr_column;
        xs[c] = xpos(0.5*(double)(frame + column_end - 1));
        frame = column_end;
    }

    for (int s = 0; s < nbr_serie; s++){
        const svg_serie_t& serie = series[s];
        if (nbr_column < 2){
            svgfile << "<circle cx=\"" << xs[0] << "\" cy=\"" << ypos(serie.column[0].avg) << "\" r=\"6\" fill=\""
                    << colors[s] << "\" />\n";
            continue;
        }
        auto points = [&](double svg_bucket_t::*bound, bool backward){
            for (int i = 0; i < nbr_column; i++){
                const int c = backward ? nbr_column-1-i : i;
                svgfile << xs[c] << "," << ypos(serie.column[c].*bound) << " ";
            }
        };
        auto stroke_mean = [&](const char* color, const char* attributes){
            svgfile << "<use xlink:href=\"#m" << s << "\" href=\"#m" << s << "\" fill=\"none\" stroke=\"" << color
                    << "\" " << attributes << " />\n";
        };

        //the mean is written once and stroked twice, a dark halo keeps it readable on top of its own band
        svgfile << "<defs><polyline id=\"m" << s << "\" points=\"";
        points(&svg_bucket_t::avg, false);
        svgfile << "\" /></defs>\n";

        if (nbr_serie == 1){ //area under the mean, only readable when nothing else is drawn over it
            svgfile << "<polygon fill=\"" << colors[s] << "\" fill-opacity=\"0.22\" points=\"" << xs[0] << "," << plot_bottom << " ";
            points(&svg_bucket_t::avg, false);
            svgfile << xs[nbr_column-1] << "," << plot_bottom << "\" />\n";
        }
        if (!is_cvvdp){ //cvvdp holds no per frame spread to show, its column value is already a pooled one
            svgfile << "<polygon fill=\"" << colors[s] << "\" fill-opacity=\"0.42\" points=\"";
            points(&svg_bucket_t::hi, false);
            points(&svg_bucket_t::lo, true);
            svgfile << "\" />\n";
        }
        stroke_mean("#000000", "stroke-opacity=\"0.6\" stroke-width=\"5.5\" stroke-linejoin=\"round\"");
        stroke_mean(colors[s], "stroke-width=\"2.2\" stroke-linejoin=\"round\" stroke-linecap=\"round\"");
    }

    rect(margin_left, margin_top, plot_width, plot_height, "fill=\"none\" stroke=\"#FFFFFF\" stroke-opacity=\"0.55\" stroke-width=\"2\"");

    group("fill=\"#FFFFFF\" font-size=\"21px\"");
    text(width*0.5, height-26., custom_indices ? "Sample" : "Frame");
    svgfile << "<g transform=\"translate(34," << margin_top + plot_height*0.5 << ") rotate(-90)\">";
    text(0., 0., is_butter ? "Distance" : "Score");
    close_group();
    close_group();

    //legend goes in the corner holding the least data
    int corner_weight[4] = {0, 0, 0, 0};
    const double middle_value = min_value + value_span*0.5;
    for (const svg_serie_t& serie: series){
        for (int c = 0; c < nbr_column; c++){
            corner_weight[((serie.column[c].avg > middle_value) ? 0 : 2) + ((2*c >= nbr_column) ? 1 : 0)]++;
        }
    }
    int corner = 0;
    for (int i = 1; i < 4; i++) if (corner_weight[i] < corner_weight[corner]) corner = i;

    std::string rows[3];
    size_t longest = 0;
    for (int s = 0; s < nbr_serie; s++){
        rows[s] = series[s].name + (is_cvvdp ? "   score " : "   mean ") + svg_num(series[s].mean)
                + (is_butter ? "   max " : "   min ") + svg_num(is_butter ? series[s].hi : series[s].lo);
        longest = std::max(longest, rows[s].size());
    }
    const double legend_width = 76. + 10.6*(double)longest;
    const double legend_height = 22. + 30.*nbr_serie;
    const double legend_x = (corner & 1) ? (plot_right - 16. - legend_width) : (margin_left + 16.);
    const double legend_y = (corner & 2) ? (plot_bottom - 16. - legend_height) : (margin_top + 16.);

    rect(legend_x, legend_y, legend_width, legend_height,
         "rx=\"6\" fill=\"#000000\" fill-opacity=\"0.78\" stroke=\"#808080\" stroke-width=\"1.5\"");
    group("fill=\"#FFFFFF\" font-size=\"19px\" text-anchor=\"start\"");
    for (int s = 0; s < nbr_serie; s++){
        const double y = legend_y + 26. + 30.*s;
        svgfile << "<line x1=\"" << legend_x+14. << "\" y1=\"" << y << "\" x2=\"" << legend_x+50. << "\" y2=\"" << y
                << "\" stroke=\"" << colors[s] << "\" stroke-width=\"4\" />\n";
        text(legend_x+62., y, rows[s], 0.35);
    }
    close_group();
    close_group();
    svgfile << "</svg>\n";

    return 0;
}
