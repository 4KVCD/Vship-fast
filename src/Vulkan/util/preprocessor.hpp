#ifndef PREPROCESSHPP
#define PREPROCESSHPP

#include <string>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include<vector>
#include <chrono>
#include <thread>
#include<exception>
#include<set>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include<cstring>
#include<algorithm>

//Our vulkanHelper will be used here using a global instance
#include "vulkanDeviceManager.hpp"
VkDeviceManagerList globalVulkanInstance; //used for the whole library

using uint = unsigned int;

#ifdef _WIN32
    #define aligned_alloc(a, b) _aligned_malloc(b, a)
#endif

#define SIGMA 1.5f
#define PI  3.14159265359
#define TAU 6.28318530718

#endif
