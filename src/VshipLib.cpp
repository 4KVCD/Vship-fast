#include "VshipColor.h"
#include "VapourSynth4.h"
#include "VSHelper4.h"

#ifndef VSHIP_VERSION_MAJOR
#define VSHIP_VERSION_MAJOR 5
#endif
#ifndef VSHIP_VERSION_MINOR
#define VSHIP_VERSION_MINOR 1
#endif
#ifndef VSHIP_VERSION_MINORMINOR
#define VSHIP_VERSION_MINORMINOR 0
#endif

#ifdef VULKANBUILD
    #include "Vulkan/butter/vapoursynth.hpp"
    #include "Vulkan/ssimu2/vapoursynth.hpp"
    #include "Vulkan/cvvdp/vapoursynth.hpp"
    #include "Vulkan/util/gpuhelper.hpp"
    #include "Vulkan/vshipGpuHelper.hpp"
#else
    #include "HIP/butter/vapoursynth.hpp"
    #include "HIP/ssimu2/vapoursynth.hpp"
    #include "HIP/cvvdp/vapoursynth.hpp"
    #include "HIP/util/gpuhelper.hpp"
    #include "HIP/vshipGpuHelper.hpp"
#endif

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(VSHIP_VERSION_MAJOR, VSHIP_VERSION_MINOR), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;qnorm:int:opt;intensity_multiplier:float:opt;distmap:int:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("CVVDP", "reference:vnode;distorted:vnode;model_name:data:opt;model_config_json:data:opt;resizeToDisplay:int:opt;distmap:int:opt;disableTemporal:int:opt;gpu_id:int:opt;", "clip:vnode;", cvvdp::CVVDPCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}

//let's define the API
#define EXPORTVSHIPLIB //to use dllexport for windows
#include "VshipAPI.h"

int global_gpu_id = 0;

enum VshipMetric{
    SSIMULACRA2,
    Butteraugli,
    CVVDP
};

struct HandlerData{
    void* implem;
    VshipMetric metric; //serves to identify the above type
    VshipError lastError = VshipError(NoError, "Your Heart", 0, "");
};

VshipError lastError = VshipError(NoError, "Your Heart", 0, "");

extern "C"{

Vship_Version Vship_GetVersion(){
    Vship_Version res;
    res.major = VSHIP_VERSION_MAJOR; res.minor = VSHIP_VERSION_MINOR; res.minorMinor = VSHIP_VERSION_MINORMINOR;
    #ifdef VULKANBUILD
    res.backend = Vship_Vulkan;
    #else
    #if defined __CUDACC__
    res.backend = Vship_Cuda;
    #else
    res.backend = Vship_HIP;
    #endif
    #endif
    return res;
}

Vship_Exception Vship_GetDeviceCount(int* number){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    *number = count;
    return Vship_NoError;
}

Vship_Exception Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id){
    try{
        Vship_GetDeviceInfoImpl(device_info, gpu_id);
    } catch (VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

Vship_Exception Vship_GPUFullCheck(int gpu_id){
    try{
        helper::gpuFullCheck(gpu_id);
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len){
    std::string cppstr = errorMessage((VSHIPEXCEPTTYPE)exception);
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    const int msglength = std::min(len-1, (int)cppstr.size());
    memcpy(out_message, cppstr.c_str(), msglength);
    out_message[msglength] = '\0'; //end character
    return cppstr.size()+1;
}

int Vship_GetDetailedLastError(char* out_message, int len){
    std::string cppstr = lastError.getErrorMessage();
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    const int msglength = std::min(len-1, (int)cppstr.size());
    memcpy(out_message, cppstr.c_str(), msglength);
    out_message[msglength] = '\0'; //end character
    return cppstr.size()+1;
}

Vship_Exception Vship_PinnedMalloc2(void** ptr, uint64_t size, int gpu_id){
    try{
        Vship_PinnedMallocImpl(ptr, size, gpu_id);
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

Vship_Exception Vship_PinnedFree2(void* ptr, int gpu_id){
    try{
        Vship_PinnedFreeImpl(ptr, gpu_id);
    } catch (const VshipError& e){
        lastError = e;
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

Vship_Exception Vship_InitHandler(void** handlerOut, void* argument){
    Vship_Exception err = Vship_NoError;

    //what type of handler do we want?
    Vship_StructType structType = *(Vship_StructType*)argument;

    try{
        switch (structType){
            case Vship_StructInitSSIMULACRA2_1:{
                auto args = *(Vship_InitSSIMULACRA2_1*)argument;
                HandlerData* handler = new HandlerData();
                handler->metric = SSIMULACRA2;
                ssimu2::SSIMU2ComputingImplementation* implem = new ssimu2::SSIMU2ComputingImplementation();
                implem->init(args.src_colorspace, args.dis_colorspace, args.gpu_id);
                handler->implem = (void*)implem;

                *((HandlerData**)handlerOut) = handler;
                break;
            }
            case Vship_StructInitButteraugli_1:{
                auto args = *(Vship_InitButteraugli_1*)argument;
                HandlerData* handler = new HandlerData();
                handler->metric = Butteraugli;
                butter::ButterComputingImplementation* implem = new butter::ButterComputingImplementation();
                implem->init(args.src_colorspace, args.dis_colorspace, args.gpu_id, args.Qnorm, args.intensity_multiplier);
                handler->implem = (void*)implem;

                *((HandlerData**)handlerOut) = handler;
                break;
            }
            case Vship_StructInitCVVDP_1:{
                auto args = *(Vship_InitCVVDP_1*)argument;
                HandlerData* handler = new HandlerData();
                handler->metric = CVVDP;
                cvvdp::CVVDPComputingImplementation* implem = new cvvdp::CVVDPComputingImplementation();
                std::string model_key;
                std::string model_config_json;
                if (args.model_key_cstr != nullptr) model_key = std::string(args.model_key_cstr);
                if (args.model_config_json_cstr != nullptr) model_config_json = std::string(args.model_config_json_cstr);
                implem->init(args.src_colorspace, args.dis_colorspace, args.gpu_id, args.fps, args.resizeToDisplay, model_key, model_config_json);
                handler->implem = (void*)implem;

                *((HandlerData**)handlerOut) = handler;
                break;
            }
            default:
                err = Vship_BadInputStructType;
                lastError = VshipError(BadInputStructType, __FILE__, __LINE__, "Used a non Init struct in the Vship C api call to create an handler");
                break;
        }
    } catch (const VshipError& e){
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_FreeHandler(void* handler){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;

    try{
        switch (actualHandler->metric){
            case SSIMULACRA2:{
                auto* implem = (ssimu2::SSIMU2ComputingImplementation*)(actualHandler->implem);
                implem->destroy();
                delete implem;
                break;
            }
            case Butteraugli:{
                auto* implem = (butter::ButterComputingImplementation*)(actualHandler->implem);
                implem->destroy();
                delete implem;
                break;
            }
            case CVVDP:{
                auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
                implem->destroy();
                delete implem;
                break;
            }
        }
    } catch (const VshipError& e){
        actualHandler->lastError = e;
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    delete actualHandler;

    return err;
}

int Vship_GetDetailedLastErrorHandler(void* handler, char* out_message, int len){
    auto* handlerdata = (HandlerData*)handler;
    std::string cppstr = handlerdata->lastError.getErrorMessage();
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    const int msglength = std::min(len-1, (int)cppstr.size());
    memcpy(out_message, cppstr.c_str(), msglength);
    out_message[msglength] = '\0'; //end character
    return cppstr.size()+1;
}

EXPORTPREPROCESS Vship_Exception Vship_ComputeHandler(void* handler, void* scoreStruct, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;
    Vship_StructType scoreStructType = *(Vship_StructType*)scoreStruct;

    try{
        switch (actualHandler->metric){
            case SSIMULACRA2:{
                if (scoreStructType != Vship_StructScoreSSIMULACRA2){
                    throw VshipError(BadInputStructType, __FILE__, __LINE__, "Used a non Init struct in the Vship C api call to retrieve score (compute)");
                }
                auto* scoreOut = (Vship_ScoreSSIMULACRA2*)scoreStruct;
                auto* implem = (ssimu2::SSIMU2ComputingImplementation*)(actualHandler->implem);
                scoreOut->score = implem->run(srcp1, srcp2, lineSize, lineSize2);
                break;
            }
            case Butteraugli:{
                if (scoreStructType != Vship_StructScoreButteraugli){
                    throw VshipError(BadInputStructType, __FILE__, __LINE__, "Used a non Init struct in the Vship C api call to retrieve score (compute)");
                }
                auto* scoreOut = (Vship_ScoreButteraugli*)scoreStruct;
                auto* implem = (butter::ButterComputingImplementation*)(actualHandler->implem);
                std::tuple<double, double, double> res = implem->run(scoreOut->dstp, scoreOut->dststride, srcp1, srcp2, lineSize, lineSize2);
                scoreOut->normQ = std::get<0>(res);
                scoreOut->norm3 = std::get<1>(res);
                scoreOut->norminf = std::get<2>(res);
                break;
            }
            case CVVDP:{
                if (scoreStructType != Vship_StructScoreCVVDP){
                    throw VshipError(BadInputStructType, __FILE__, __LINE__, "Used a non Init struct in the Vship C api call to retrieve score (compute)");
                }
                auto* scoreOut = (Vship_ScoreCVVDP*)scoreStruct;
                auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
                scoreOut->score = implem->run(scoreOut->dstp, scoreOut->dststride, srcp1, srcp2, lineSize, lineSize2);
                break;
            }
        }
    } catch (const VshipError& e){
        actualHandler->lastError = e;
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

EXPORTPREPROCESS Vship_Exception Vship_GetDistmapResolution(void* handler, int64_t* width, int64_t* height){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;

    switch (actualHandler->metric){
        case Butteraugli:{
            auto* implem = (butter::ButterComputingImplementation*)(actualHandler->implem);
            auto answerPair = implem->getDistmapResolution();
            *width = answerPair.first;
            *height = answerPair.second;
            break;
        }
        case CVVDP:{
            auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
            *width = implem->resize_width;
            *height = implem->resize_height;
            break;
        }
        default:{
            err = Vship_BadHandler;
            lastError = VshipError(BadHandler, __FILE__, __LINE__, "Tried to retrieve distmapResolution on a metric that does not possess a distmap in Vship C api");
            break;
        }
    }
    return err;
}

EXPORTPREPROCESS Vship_Exception Vship_Reset(void* handler){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;

    try{
        switch (actualHandler->metric){
            case CVVDP:{
                auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
                implem->flushTemporalRing();
                break;
            }
            default:
                throw VshipError(BadHandler, __FILE__, __LINE__, "Reset does not work on non temporal metric");
        }
    } catch (const VshipError& e){
        actualHandler->lastError = e;
        lastError = e;
        err = (Vship_Exception)e.type;
    }

    return err;
}

EXPORTPREPROCESS Vship_Exception Vship_ResetScore(void* handler){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;

    try{
        switch (actualHandler->metric){
            case CVVDP:{
                auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
                implem->flushOnlyScore();
                break;
            }
            default:
                throw VshipError(BadHandler, __FILE__, __LINE__, "Reset does not work on non temporal metric");
        }
    } catch (const VshipError& e){
        actualHandler->lastError = e;
        lastError = e;
        err = (Vship_Exception)e.type;
    }

    return err;
}

//This function allows loading frames to the temporal filter to represent past frames without impacting the score for CVVDP
EXPORTPREPROCESS Vship_Exception Vship_LoadTemporalCVVDP2(void* handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    auto* actualHandler = (HandlerData*)handler;

    try{
        switch (actualHandler->metric){
            case CVVDP:{
                auto* implem = (cvvdp::CVVDPComputingImplementation*)(actualHandler->implem);
                implem->loadImageToRing(srcp1, srcp2, lineSize, lineSize2);
                break;
            }
            default:
                throw VshipError(BadHandler, __FILE__, __LINE__, "Reset does not work on non temporal metric");
        }
    } catch (const VshipError& e){
        actualHandler->lastError = e;
        lastError = e;
        err = (Vship_Exception)e.type;
    }
    return err;
}

//------------------------------------------------------------------
//DEPRECATION WARNING: everything after this mark only constitutes backward compatibility
//------------------------------------------------------------------

Vship_Exception Vship_SetDevice(int gpu_id){
    int numgpu;
    Vship_Exception errcount = Vship_GetDeviceCount(&numgpu);
    if (errcount != Vship_NoError){
        return errcount;
    }
    if (gpu_id >= numgpu){
        lastError = VshipError(BadDeviceArgument, __FILE__, __LINE__, "In Vship_SetDevice");
        return Vship_BadDeviceArgument;
    }
    global_gpu_id = gpu_id;
    return Vship_NoError;
}

Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size){
    return Vship_PinnedMalloc2(ptr, size, global_gpu_id);
}

Vship_Exception Vship_PinnedFree(void* ptr){
    return Vship_PinnedFree2(ptr, global_gpu_id);
}

//to mimic old behavior with integer as handlers
RessourceManager<void*> handlerHolder;

Vship_Exception Vship_SSIMU2Init2(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int gpu_id){
    Vship_InitSSIMULACRA2_1 initStruct;
    initStruct.structType = Vship_StructInitSSIMULACRA2_1;
    initStruct.src_colorspace = src_colorspace;
    initStruct.dis_colorspace = dis_colorspace;
    initStruct.gpu_id = gpu_id;

    void* internalhandler;
    Vship_Exception err = Vship_InitHandler(&internalhandler, (void*)&initStruct);
    if (err != Vship_NoError) return err;

    handler->id = handlerHolder.allocate();
    handlerHolder.lock.lock();
    handlerHolder.elements[handler->id] = internalhandler;
    handlerHolder.lock.unlock();
    return err;
}

Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace){
    return Vship_SSIMU2Init2(handler, src_colorspace, dis_colorspace, global_gpu_id);
}

Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    auto* handlerdata = handlerHolder.elements[handler.id];
    err = Vship_FreeHandler(handlerdata);
    handlerHolder.lock.unlock();
    handlerHolder.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeSSIMU2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeSSIMU2 is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();

    Vship_ScoreSSIMULACRA2 scoreStruct;
    scoreStruct.structType = Vship_StructScoreSSIMULACRA2;

    err = Vship_ComputeHandler(handlerdata, (void*)&scoreStruct, srcp1, srcp2, lineSize, lineSize2);
    *score = scoreStruct.score;
    return err;
}

int Vship_SSIMU2GetDetailedLastError(Vship_SSIMU2Handler handler, char* out_message, int len){
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_SSIMU2GetDetailedLastError is not valid, did you allocate or use after free?");
        return 0;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    return Vship_GetDetailedLastErrorHandler(handlerdata, out_message, len);
}

Vship_Exception Vship_ButteraugliInit2(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier, int gpu_id){
    Vship_InitButteraugli_1 initStruct;
    initStruct.structType = Vship_StructInitButteraugli_1;
    initStruct.src_colorspace = src_colorspace;
    initStruct.dis_colorspace = dis_colorspace;
    initStruct.Qnorm = Qnorm;
    initStruct.intensity_multiplier = intensity_multiplier;
    initStruct.gpu_id = gpu_id;

    void* internalhandler;
    Vship_Exception err = Vship_InitHandler(&internalhandler, (void*)&initStruct);
    if (err != Vship_NoError) return err;

    handler->id = handlerHolder.allocate();
    handlerHolder.lock.lock();
    handlerHolder.elements[handler->id] = internalhandler;
    handlerHolder.lock.unlock();
    return err;
}

Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, int Qnorm, float intensity_multiplier){
    return Vship_ButteraugliInit2(handler, src_colorspace, dis_colorspace, Qnorm, intensity_multiplier, global_gpu_id);
}

Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    auto* handlerdata = handlerHolder.elements[handler.id];
    err = Vship_FreeHandler(handlerdata);
    handlerHolder.lock.unlock();
    handlerHolder.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeButteraugli(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeButteraugli is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();

    Vship_ScoreButteraugli scoreStruct;
    scoreStruct.structType = Vship_StructScoreButteraugli;
    scoreStruct.dstp = dstp;
    scoreStruct.dststride = dststride;

    err = Vship_ComputeHandler(handlerdata, (void*)&scoreStruct, srcp1, srcp2, lineSize, lineSize2);
    score->normQ = scoreStruct.normQ;
    score->norm3 = scoreStruct.norm3;
    score->norminf = scoreStruct.norminf;
    return err;
}

int Vship_ButteraugliGetDetailedLastError(Vship_ButteraugliHandler handler, char* out_message, int len){
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ButteraugliGetDetailedLastError is not valid, did you allocate or use after free?");
        return 0;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    return Vship_GetDetailedLastErrorHandler(handlerdata, out_message, len);
}

Vship_Exception Vship_CVVDPInit3(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr, const char* model_config_json_cstr, int gpu_id){
    Vship_InitCVVDP_1 initStruct;
    initStruct.structType = Vship_StructInitCVVDP_1;
    initStruct.src_colorspace = src_colorspace;
    initStruct.dis_colorspace = dis_colorspace;
    initStruct.fps = fps;
    initStruct.resizeToDisplay = resizeToDisplay;
    initStruct.model_key_cstr = model_key_cstr;
    initStruct.model_config_json_cstr = model_config_json_cstr;
    initStruct.gpu_id = gpu_id;

    void* internalhandler;
    Vship_Exception err = Vship_InitHandler(&internalhandler, (void*)&initStruct);
    if (err != Vship_NoError) return err;

    handler->id = handlerHolder.allocate();
    handlerHolder.lock.lock();
    handlerHolder.elements[handler->id] = internalhandler;
    handlerHolder.lock.unlock();
    return err;
}

Vship_Exception Vship_CVVDPInit2(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr, const char* model_config_json_cstr){
    return Vship_CVVDPInit3(handler, src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key_cstr, model_config_json_cstr, global_gpu_id);
}

Vship_Exception Vship_CVVDPInit(Vship_CVVDPHandler* handler, Vship_Colorspace_t src_colorspace, Vship_Colorspace_t dis_colorspace, float fps, bool resizeToDisplay, const char* model_key_cstr){
    return Vship_CVVDPInit2(handler, src_colorspace, dis_colorspace, fps, resizeToDisplay, model_key_cstr, NULL);
}

Vship_Exception Vship_CVVDPFree(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    auto* handlerdata = handlerHolder.elements[handler.id];
    err = Vship_FreeHandler(handlerdata);
    handlerHolder.lock.unlock();
    handlerHolder.free(handler.id);
    return err;
}

Vship_Exception Vship_CVVDPGetDistmapResolution(Vship_CVVDPHandler handler, int64_t* width, int64_t* height){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_CVVDPGetDistmapResolution is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();

    err = Vship_GetDistmapResolution(handlerdata, width, height);
    return err;
}

Vship_Exception Vship_ResetCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ResetCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    err = Vship_Reset(handlerdata);
    return err;
}

//The temporalFilter stays loaded but the score goes back to neutral
Vship_Exception Vship_ResetScoreCVVDP(Vship_CVVDPHandler handler){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ResetScoreCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    err = Vship_ResetScore(handlerdata);
    return err;
}

Vship_Exception Vship_LoadTemporalCVVDP(Vship_CVVDPHandler handler, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_LoadTemporalCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    err = Vship_LoadTemporalCVVDP2(handlerdata, srcp1, srcp2, lineSize, lineSize2);
    return err;
}


Vship_Exception Vship_ComputeCVVDP(Vship_CVVDPHandler handler, double* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], const int64_t lineSize[3], const int64_t lineSize2[3]){
    Vship_Exception err = Vship_NoError;
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_ComputeCVVDP is not valid, did you allocate or use after free?");
        return Vship_BadHandler;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    handlerHolder.lock.unlock();

    Vship_ScoreCVVDP scoreStruct;
    scoreStruct.structType = Vship_StructScoreCVVDP;
    scoreStruct.dstp = dstp;
    scoreStruct.dststride = dststride;

    err = Vship_ComputeHandler(handlerdata, (void*)&scoreStruct, srcp1, srcp2, lineSize, lineSize2);
    *score = scoreStruct.score;
    return err;
}

int Vship_CVVDPGetDetailedLastError(Vship_CVVDPHandler handler, char* out_message, int len){
    handlerHolder.lock.lock();
    if (handler.id >= handlerHolder.elements.size()){
        handlerHolder.lock.unlock();
        lastError = VshipError(BadHandler, __FILE__, __LINE__, "Handler internal state in Vship_CVVDPGetDetailedLastError is not valid, did you allocate or use after free?");
        return 0;
    }
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    auto* handlerdata = handlerHolder.elements[handler.id];
    return Vship_GetDetailedLastErrorHandler(handlerdata, out_message, len);
}

} //extern "C"
