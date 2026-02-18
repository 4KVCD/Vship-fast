#pragma once

namespace cvvdp{

constexpr int pu_dilate = 3;
constexpr int beta = 2;
constexpr int beta_t = 2;
constexpr int beta_tch = 4;
constexpr int beta_sch = 4;
constexpr float csf_sigma = -1.5;
constexpr float sensitivity_correction = -0.2797423303127289;
constexpr float jod_a = 0.0439569391310215;
constexpr float jod_exp = 0.9302042722702026;
constexpr int filter_len = -1;
constexpr float sigma_tf[4] = {
    5.79336,
    14.1255,
    6.63661,
    0.12314,
};
constexpr float beta_tf[4] = {
    1.3314,
    1.1196,
    0.947901,
    0.1898
};
//constexpr bool xchannel_masking = true;
constexpr float image_int = 0.577918291091919;
constexpr float bfilt_duration = 0.4;

}