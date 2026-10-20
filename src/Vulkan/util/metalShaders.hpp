#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

//The shaders as MoltenVK (Vulkan on Metal, Apple's GPUs) runs them, made from the embedded ones when they are loaded.
//
//Every float operation of the embedded shaders is decorated NoContraction (spvFileToCppHeader.cpp), so that no GPU
//compiler fuses a multiplication and an addition into one rounding. SPIRV-Cross, which MoltenVK translates SPIR-V to
//Metal with, honours it by calling an [[clang::optnone]] function for each decorated multiplication, addition and
//subtraction (spvFMul(a, b) = fma(a, b, 0) ...): 5681 calls in our shaders, which made the metrics 2-5 times slower on
//an M1 Pro, and Metal's fast math, which MoltenVK compiles with by default, broke CVVDP's scores on top of it. Metal
//itself fuses a*b + c in all of its math modes, so leaving the decorations out is not an option either.
//
//So for MoltenVK the decorations are removed and every float product a*b becomes fma(a, b, z), with z a -0.0 the host
//pushes after the kernel's arguments (negativeZeroOffset). fma(a, b, -0.0) is a*b rounded once, with a*b's sign of
//zero; z is only known when the kernel runs, so no compiler can turn it back into a multiplication, and with no
//multiplication left nothing is fused with an addition. The GPU's fma costs what its multiplication does.
//
//Metal's fast math is also turned off, per kernel, with SPIR-V's SignedZeroInfNanPreserve execution mode (MoltenVK
//then compiles that kernel in Metal's safe math mode): fast math reassociates sums, which Butteraugli's double-float
//arithmetic and the correctly rounded divide() must not have. It stays on for CVVDP's kernels and for YUVToLinRGB:
//their results go through pow, exp and log, whose fast versions are closer to NVIDIA's and AMD's (on a synthetic
//clip, CVVDP within 2 ulp of an RTX 3070 with it, 3 without; Butteraugli on YUV input within the NVIDIA-AMD gap), and
//CVVDP is 18% faster with it.
//
//Other GPUs load the embedded shaders unchanged.
namespace metalShaders {

//where the host pushes z, after the 128 bytes of kernel arguments
constexpr uint32_t negativeZeroOffset = 128;

//whether a shader keeps Metal's fast math (see above)
inline bool keepsFastMath(const std::string& shaderName){
    return shaderName.find("CVVDP") != std::string::npos || shaderName == "YUVToLinRGB.spv";
}

namespace detail {

constexpr uint32_t opExtension = 10, opExtInstImport = 11, opExtInst = 12, opMemoryModel = 14, opEntryPoint = 15,
    opExecutionMode = 16, opCapability = 17, opTypeInt = 21, opTypeFloat = 22, opTypeVector = 23, opTypeStruct = 30,
    opTypePointer = 32, opConstant = 43, opFunction = 54, opFunctionParameter = 55, opLabel = 248, opVariable = 59,
    opLoad = 61, opAccessChain = 65, opInBoundsAccessChain = 66, opDecorate = 71, opMemberDecorate = 72,
    opCompositeConstruct = 80, opFMul = 133, opVectorTimesScalar = 142, opMatrixTimesScalar = 143,
    opVectorTimesMatrix = 144, opMatrixTimesVector = 145, opMatrixTimesMatrix = 146, opDot = 148;
constexpr uint32_t decorationNoContraction = 42, decorationOffset = 35, storagePushConstant = 9, glslFma = 50,
    capabilitySignedZeroInfNanPreserve = 4466, executionModeSignedZeroInfNanPreserve = 4461;

using Instruction = std::vector<uint32_t>;

inline uint32_t opcode(const Instruction& i){ return i[0] & 0xffff; }

inline Instruction make(uint32_t op, std::vector<uint32_t> operands){
    operands.insert(operands.begin(), (uint32_t)(operands.size() + 1) << 16 | op);
    return operands;
}

inline std::vector<uint32_t> stringWords(const std::string& s){
    std::vector<uint32_t> words(s.size()/4 + 1, 0);
    std::memcpy(words.data(), s.data(), s.size());
    return words;
}

inline std::string wordsString(const uint32_t* words, size_t count){
    std::string s((const char*)words, count*4);
    return s.substr(0, s.find('\0'));
}

}

//The MoltenVK version of an embedded shader (see above). The input is valid SPIR-V from slangc: one push-constant
//block per entry point, every function an entry point (slangc inlines the others).
inline std::vector<char> rewrite(const std::vector<char>& embedded, bool fastMath){
    using namespace detail;
    const uint32_t* words = (const uint32_t*)embedded.data();
    const size_t wordCount = embedded.size()/4;
    uint32_t bound = words[3];
    auto newId = [&]{ return bound++; };

    std::vector<Instruction> module;
    for (size_t i = 5; i < wordCount; i += words[i] >> 16) module.emplace_back(words + i, words + i + (words[i] >> 16));
    module.erase(std::remove_if(module.begin(), module.end(), [](const Instruction& i){
        return opcode(i) == opDecorate && i.size() >= 3 && i[2] == decorationNoContraction;
    }), module.end());

    //the types and variables involved
    uint32_t float32 = 0, int32 = 0, glsl = 0, pushFloatPointer = 0;
    std::unordered_map<uint32_t, uint32_t> floatVectorSize; //vector type -> component count
    std::unordered_map<uint32_t, uint32_t> pointee;         //pointer type -> pointed type
    std::unordered_map<uint32_t, uint32_t> pushBlock;       //push-constant variable -> its struct type
    for (const Instruction& i : module){
        if (opcode(i) == opTypeFloat && i[2] == 32) float32 = i[1];
        if (opcode(i) == opTypeInt && i[2] == 32 && i[3] == 1) int32 = i[1];
        if (opcode(i) == opExtInstImport && wordsString(&i[2], i.size() - 2) == "GLSL.std.450") glsl = i[1];
        if (opcode(i) == opTypePointer) pointee[i[1]] = i[3];
        if (opcode(i) == opTypePointer && i[2] == storagePushConstant && i[3] == float32 && float32 != 0) pushFloatPointer = i[1];
        if (opcode(i) == opVariable && i[3] == storagePushConstant) pushBlock[i[2]] = pointee[i[1]];
    }
    for (const Instruction& i : module) if (opcode(i) == opTypeVector && i[2] == float32) floatVectorSize[i[1]] = i[3];
    if (float32 == 0 || pushBlock.empty()) return embedded;

    std::vector<Instruction> imports, annotations, globals, capabilities, extensions, modes;
    if (glsl == 0){
        glsl = newId();
        std::vector<uint32_t> operands = {glsl};
        for (uint32_t w : stringWords("GLSL.std.450")) operands.push_back(w);
        imports.push_back(make(opExtInstImport, operands));
    }
    if (int32 == 0){ int32 = newId(); globals.push_back(make(opTypeInt, {int32, 32, 1})); }
    if (pushFloatPointer == 0){
        pushFloatPointer = newId();
        globals.push_back(make(opTypePointer, {pushFloatPointer, storagePushConstant, float32}));
    }

    //z, a float member at negativeZeroOffset appended to every push-constant block
    std::unordered_map<uint32_t, uint32_t> zMemberIndex; //struct type -> constant id of z's member index
    for (Instruction& i : module){
        if (opcode(i) != opTypeStruct || zMemberIndex.count(i[1])) continue;
        bool isPushBlock = false;
        for (auto& [variable, block] : pushBlock) isPushBlock |= block == i[1];
        if (!isPushBlock) continue;
        //z's type must be declared before the block it is added to
        size_t floatAt = 0, blockAt = 0;
        for (size_t j = 0; j < module.size(); j++){
            if (module[j].size() > 1 && module[j][1] == float32 && opcode(module[j]) == opTypeFloat) floatAt = j;
            if (&module[j] == &i) blockAt = j;
        }
        if (floatAt > blockAt) return embedded;
        const uint32_t member = (uint32_t)i.size() - 2;
        i.push_back(float32);
        i[0] = (uint32_t)i.size() << 16 | opTypeStruct;
        annotations.push_back(make(opMemberDecorate, {i[1], member, decorationOffset, negativeZeroOffset}));
        const uint32_t index = newId();
        globals.push_back(make(opConstant, {int32, index, member}));
        zMemberIndex[i[1]] = index;
    }

    //the functions: z loaded first, then every float product turned into fma(a, b, z)
    size_t firstFunction = 0;
    while (firstFunction < module.size() && opcode(module[firstFunction]) != opFunction) firstFunction++;
    std::vector<Instruction> functions;
    for (size_t f = firstFunction; f < module.size();){
        size_t end = f + 1;
        while (end < module.size() && opcode(module[end]) != opFunction) end++;
        //this function's push-constant block: the one it reads, any if it reads none (it then only needs z)
        uint32_t block = pushBlock.begin()->first;
        for (size_t k = f; k < end; k++){
            const Instruction& i = module[k];
            if ((opcode(i) == opAccessChain || opcode(i) == opInBoundsAccessChain) && pushBlock.count(i[3])) { block = i[3]; break; }
        }
        size_t start = f + 1;
        while (opcode(module[start]) == opFunctionParameter) start++;
        start++; //OpLabel
        while (opcode(module[start]) == opVariable) start++;
        const uint32_t zPointer = newId(), z = newId();
        std::vector<Instruction> prologue = {
            make(opAccessChain, {pushFloatPointer, zPointer, block, zMemberIndex[pushBlock[block]]}),
            make(opLoad, {float32, z, zPointer}),
        };
        std::unordered_map<uint32_t, uint32_t> zOfType = {{float32, z}};
        auto zFor = [&](uint32_t type){
            if (!zOfType.count(type)){
                const uint32_t splat = newId();
                prologue.push_back(make(opCompositeConstruct, std::vector<uint32_t>{type, splat}));
                for (uint32_t c = 0; c < floatVectorSize[type]; c++) prologue.back().push_back(z);
                prologue.back()[0] = (uint32_t)prologue.back().size() << 16 | opCompositeConstruct;
                zOfType[type] = splat;
            }
            return zOfType[type];
        };
        std::vector<Instruction> body;
        for (size_t k = f; k < end; k++){
            const Instruction& i = module[k];
            const bool isFloat = i.size() > 1 && (i[1] == float32 || floatVectorSize.count(i[1]));
            if (opcode(i) == opFMul && isFloat){
                body.push_back(make(opExtInst, {i[1], i[2], glsl, glslFma, i[3], i[4], zFor(i[1])}));
            } else if (opcode(i) == opVectorTimesScalar && isFloat){
                const uint32_t splat = newId();
                Instruction construct = make(opCompositeConstruct, {i[1], splat});
                for (uint32_t c = 0; c < floatVectorSize[i[1]]; c++) construct.push_back(i[4]);
                construct[0] = (uint32_t)construct.size() << 16 | opCompositeConstruct;
                body.push_back(construct);
                body.push_back(make(opExtInst, {i[1], i[2], glsl, glslFma, i[3], splat, zFor(i[1])}));
            } else {
                //products of matrices and dot products would need expanding; our shaders have none
                if (isFloat && (opcode(i) == opMatrixTimesScalar || opcode(i) == opVectorTimesMatrix ||
                    opcode(i) == opMatrixTimesVector || opcode(i) == opMatrixTimesMatrix || opcode(i) == opDot)) return embedded;
                body.push_back(i);
            }
        }
        body.insert(body.begin() + (start - f), prologue.begin(), prologue.end());
        functions.insert(functions.end(), body.begin(), body.end());
        f = end;
    }

    if (!fastMath){
        capabilities.push_back(make(opCapability, {capabilitySignedZeroInfNanPreserve}));
        extensions.push_back(make(opExtension, stringWords("SPV_KHR_float_controls")));
        for (const Instruction& i : module)
            if (opcode(i) == opEntryPoint) modes.push_back(make(opExecutionMode, {i[2], executionModeSignedZeroInfNanPreserve, 32}));
    }

    //reassembled in SPIR-V's section order
    std::vector<uint32_t> out(words, words + 5);
    auto emit = [&](const std::vector<Instruction>& list){ for (const Instruction& i : list) out.insert(out.end(), i.begin(), i.end()); };
    auto inSection = [&](uint32_t op, std::initializer_list<uint32_t> ops){ for (uint32_t o : ops) if (o == op) return true; return false; };
    size_t k = 0;
    auto emitWhile = [&](std::initializer_list<uint32_t> ops){
        while (k < firstFunction && inSection(opcode(module[k]), ops)) emit({module[k++]});
    };
    emitWhile({opCapability});
    emit(capabilities);
    emitWhile({opExtension});
    emit(extensions);
    emitWhile({opExtInstImport});
    emit(imports);
    emitWhile({opMemoryModel, opEntryPoint, opExecutionMode, 331 /*OpExecutionModeId*/});
    emit(modes);
    //debug instructions and annotations, then the types, constants and global variables
    size_t lastAnnotation = k;
    for (size_t j = k; j < firstFunction; j++) if (inSection(opcode(module[j]), {opDecorate, opMemberDecorate, 332, 5632, 5633})) lastAnnotation = j + 1;
    for (; k < lastAnnotation; k++) emit({module[k]});
    emit(annotations);
    for (; k < firstFunction; k++) emit({module[k]});
    emit(globals);
    emit(functions);
    out[3] = bound;

    std::vector<char> result(out.size()*4);
    std::memcpy(result.data(), out.data(), result.size());
    return result;
}

}
