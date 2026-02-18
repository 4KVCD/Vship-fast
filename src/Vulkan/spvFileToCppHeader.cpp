#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>
#include <cstdint>
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
        std::vector<char> bytecode(fileSize);
        file.seekg(0);
        file.read(bytecode.data(), fileSize);
        file.close();
        out << "{\"" << shaderName << "\", {";
        for (size_t i = 0; i < bytecode.size(); i++){
            out << (int)bytecode[i] << ",";
        }
        out << "}}," << std::endl;
    }
    out << "};" << std::endl;
}