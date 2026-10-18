namespace ssimu2{

//SSIMULACRA2 blurs with libjxl's recursive Gaussian (tools/gauss_blur.cc, "Recursive Implementation of the
//Gaussian Filter Using Truncated Cosine Functions", Charalampidis 2016) of sigma 1.5, zero outside the image.
//GaussianHandle computes its coefficients like libjxl's CreateRecursiveGaussian, in double rounded to float, so
//the shaders (shaders/scoreSSIMU2.slang) run the recurrence on the same numbers.
constexpr int recursiveGaussianRadius = 5; //round(3.2795*1.5 + 0.2546), RADIUS in the shaders
//Layout of the coefficient buffer: the 3 terms (k = 1, 3, 5 of the paper) of mul_in, mul_prev and mul_prev2
//with 4 lanes each (the horizontal pass computes 4 outputs at a time like libjxl), then n2 and d1.
constexpr int recursiveGaussianCoefficients = 3*12 + 3 + 3;

//Inverts a 3x3 matrix in place like libjxl's Inv3x3Matrix (lib/jxl/base/matrix_ops.h).
inline void invert3x3(double m[3][3]){
    double t[3][3];
    t[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    t[0][1] = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    t[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    t[1][0] = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    t[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    t[1][2] = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    t[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    t[2][1] = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    t[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    const double det = m[0][0] * t[0][0] + m[0][1] * t[1][0] + m[0][2] * t[2][0];
    const double idet = 1.0 / det;
    for (int j = 0; j < 3; j++){
        for (int i = 0; i < 3; i++){
            m[j][i] = t[j][i] * idet;
        }
    }
}

//libjxl's CreateRecursiveGaussian, written into out with the layout above. Returns the radius.
inline int computeRecursiveGaussian(double sigma, float* out){
    const double kPi = 3.141592653589793238;
    const double radius = std::round(3.2795 * sigma + 0.2546);

    const double pi_div_2r = kPi / (2.0 * radius);
    const double omega[3] = {pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r};

    const double p_1 = +1.0 / std::tan(0.5 * omega[0]);
    const double p_3 = -1.0 / std::tan(0.5 * omega[1]);
    const double p_5 = +1.0 / std::tan(0.5 * omega[2]);

    const double r_1 = +p_1 * p_1 / std::sin(omega[0]);
    const double r_3 = -p_3 * p_3 / std::sin(omega[1]);
    const double r_5 = +p_5 * p_5 / std::sin(omega[2]);

    const double neg_half_sigma2 = -0.5 * sigma * sigma;
    const double recip_radius = 1.0 / radius;
    double rho[3];
    for (int i = 0; i < 3; i++){
        rho[i] = std::exp(neg_half_sigma2 * omega[i] * omega[i]) * recip_radius;
    }

    const double D_13 = p_1 * r_3 - r_1 * p_3;
    const double D_35 = p_3 * r_5 - r_3 * p_5;
    const double D_51 = p_5 * r_1 - r_5 * p_1;

    const double recip_d13 = 1.0 / D_13;
    const double zeta_15 = D_35 * recip_d13;
    const double zeta_35 = D_51 * recip_d13;

    double A[3][3] = {{p_1, p_3, p_5}, {r_1, r_3, r_5}, {zeta_15, zeta_35, 1}};
    invert3x3(A);
    const double gamma[3] = {1, radius * radius - sigma * sigma, zeta_15 * rho[0] + zeta_35 * rho[1] + rho[2]};
    double beta[3];
    for (int y = 0; y < 3; y++){
        double e = 0;
        for (int x = 0; x < 3; x++){
            e += A[y][x] * gamma[x];
        }
        beta[y] = e;
    }

    float* mul_in = out;
    float* mul_prev = out + 12;
    float* mul_prev2 = out + 24;
    float* n2_out = out + 36;
    float* d1_out = out + 39;
    for (int i = 0; i < 3; i++){
        const double n2 = -beta[i] * std::cos(omega[i] * (radius + 1.0));
        const double d1 = -2.0 * std::cos(omega[i]);
        n2_out[i] = (float)n2;
        d1_out[i] = (float)d1;

        const double d_2 = d1 * d1;
        mul_prev[4 * i + 0] = -d1;
        mul_prev[4 * i + 1] = d_2 - 1.0;
        mul_prev[4 * i + 2] = -d_2 * d1 + 2.0 * d1;
        mul_prev[4 * i + 3] = d_2 * d_2 - 3.0 * d_2 + 1.0;
        mul_prev2[4 * i + 0] = -1.0;
        mul_prev2[4 * i + 1] = d1;
        mul_prev2[4 * i + 2] = -d_2 + 1.0;
        mul_prev2[4 * i + 3] = d_2 * d1 - 2.0 * d1;
        mul_in[4 * i + 0] = n2;
        mul_in[4 * i + 1] = -d1 * n2;
        mul_in[4 * i + 2] = d_2 * n2 - n2;
        mul_in[4 * i + 3] = -d_2 * d1 * n2 + 2.0 * d1 * n2;
    }
    return (int)radius;
}

class GaussianHandle{
public:
    VshipDeviceHelper* deviceHelper;
    VkDeviceAddress coefficients_d = 0;
    float* coefficients;
    //this cmdbuff needs to be executed as pretreatment at init
    void init(VshipDeviceHelper* deviceHelper, VkCommandBuffer cmdbuff){
        this->deviceHelper = deviceHelper;
        coefficients = (float*)deviceHelper->device.mallocHost(recursiveGaussianCoefficients*sizeof(float));
        const int radius = computeRecursiveGaussian(1.5, coefficients);
        assert(radius == recursiveGaussianRadius);
        (void)radius;
        coefficients_d = deviceHelper->device.mallocDevice(recursiveGaussianCoefficients*sizeof(float));
        deviceHelper->device.memCpyHtoD(cmdbuff, coefficients_d, coefficients, recursiveGaussianCoefficients*sizeof(float));
    }
    void destroy(){
        deviceHelper->device.freeHost(coefficients);
        deviceHelper->device.freeDevice(coefficients_d);
    }
};

}
