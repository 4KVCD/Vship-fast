#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

//this file embed all spv files in a directory to a unique .hpp file with the given structure:
//#pragma once
//#include<string>
//#include<unordered_map>
//#include<vector>
//
//std::unordered_map<std::string, std::vector<char>> shaderNameToByteCode = {
//...
//};
//
//Each shader gets the NoContraction decoration on every floating point addition, subtraction, multiplication,
//division and product before it is embedded. Without it, the GPU's compiler may fuse a multiplication and an
//addition into one fused multiply-add, rounded once instead of twice, wherever it likes; NVIDIA's and AMD's
//compilers choose different places, and the places change when the code around them is restructured. With it,
//every operation is rounded on its own, so a shader gives the same floats on every vendor and whatever kernels its
//operations are split into. A fused multiply-add the metric wants is written as an explicit fma() in the shader,
//which this does not touch. Slang does not emit the decoration itself: it ignores the precise keyword.

namespace {

constexpr uint32_t spirvMagic = 0x07230203;
constexpr uint32_t opDecorate = 71;
constexpr uint32_t decorationNoContraction = 42;

//the operations a compiler could combine with another into one with a single rounding
const std::unordered_set<uint32_t> contractibleOps = {
    129, //OpFAdd
    131, //OpFSub
    133, //OpFMul
    136, //OpFDiv
    142, //OpVectorTimesScalar
    143, //OpMatrixTimesScalar
    144, //OpVectorTimesMatrix
    145, //OpMatrixTimesVector
    146, //OpMatrixTimesMatrix
    147, //OpOuterProduct
    148, //OpDot
};

//the instructions of a module that come before its annotations end (capabilities to decorations); the decorations
//are inserted before the first instruction that is none of these
const std::unordered_set<uint32_t> headerOps = {
    17, 10, 11, 14, 15, 16, 331, //OpCapability, OpExtension, OpExtInstImport, OpMemoryModel, OpEntryPoint, OpExecutionMode(Id)
    7, 2, 3, 4, 5, 6, 330,       //OpString, OpSourceContinued, OpSource, OpSourceExtension, OpName, OpMemberName, OpModuleProcessed
    71, 72, 73, 74, 75, 332, 5632, 5633, //OpDecorate, OpMemberDecorate, decoration groups, OpDecorateId, OpDecorateString, OpMemberDecorateString
    8, 317,                      //OpLine, OpNoLine
};

//Adds NoContraction to the result of every contractible operation of the module in words (see above).
bool addNoContraction(std::vector<uint32_t>& words){
    if (words.size() < 5 || words[0] != spirvMagic) return false;
    std::unordered_set<uint32_t> decorated; //results that already have it
    std::vector<uint32_t> results;
    size_t insertAt = 0;
    for (size_t i = 5; i < words.size();){
        const uint32_t wordCount = words[i] >> 16;
        const uint32_t opcode = words[i] & 0xffff;
        if (wordCount == 0 || i + wordCount > words.size()) return false;
        if (insertAt == 0 && headerOps.count(opcode) == 0) insertAt = i;
        if (opcode == opDecorate && wordCount >= 3 && words[i+2] == decorationNoContraction) decorated.insert(words[i+1]);
        if (contractibleOps.count(opcode) != 0 && wordCount >= 3) results.push_back(words[i+2]); //result type, result id, operands
        i += wordCount;
    }
    if (insertAt == 0) return false;
    std::vector<uint32_t> decorations;
    for (uint32_t id : results){
        if (decorated.count(id) != 0) continue;
        decorations.push_back((3u << 16) | opDecorate);
        decorations.push_back(id);
        decorations.push_back(decorationNoContraction);
    }
    words.insert(words.begin() + insertAt, decorations.begin(), decorations.end());
    return true;
}

}

int main(int argc, char* argv[]){
    //first argument is input directory
    //second argument is outputFile
    if (argc != 3) return 1;
    std::filesystem::path inputDirectory = argv[1];
    std::filesystem::path outputFile = argv[2];

    //std::cout << inputDirectory << " " << outputFile << std::endl;
    std::ofstream out(outputFile);
    if (!out) return 1;
    out << "#pragma once" << std::endl;
    out << "#include<string>" << std::endl;
    out << "#include<unordered_map>" << std::endl;
    out << "#include<vector>" << std::endl;
    out << std::endl;
    out << "std::unordered_map<std::string, std::vector<char>> shaderNameToByteCode = {" << std::endl;
    for (const auto & entry : std::filesystem::directory_iterator(inputDirectory)){
        std::string shaderName = entry.path().filename().string();
        if (shaderName == ".gitkeep") continue;
        std::ifstream file(entry.path(), std::ios::ate | std::ios_base::binary);
        if (!file) return 1;
        size_t fileSize = (size_t)file.tellg();
        if (fileSize % 4 != 0){
            std::cerr << shaderName << ": not a SPIR-V module (size not a multiple of 4)" << std::endl;
            return 1;
        }
        std::vector<uint32_t> words(fileSize/4);
        file.seekg(0);
        file.read((char*)words.data(), fileSize);
        file.close();
        if (!addNoContraction(words)){
            std::cerr << shaderName << ": not a valid SPIR-V module" << std::endl;
            return 1;
        }
        std::vector<char> bytecode(words.size()*4);
        std::memcpy(bytecode.data(), words.data(), bytecode.size());
        out << "{\"" << shaderName << "\", {";
        for (size_t i = 0; i < bytecode.size(); i++){
            out << (int)bytecode[i] << ",";
        }
        out << "}}," << std::endl;
    }
    out << "};" << std::endl;
}
