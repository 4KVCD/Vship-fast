#pragma once

namespace cvvdp{

//for now only accepts squares Tiling (see GaussianSmart_device out and tot arrays)
//only allows powers of 2 for Thx and Thy due to pointer jumping (8/8 16/16 and 32/32)
constexpr int gaussianThx = 16;
constexpr int gaussianThy = 16;
//Optimization Island in gaussianSize lies in gaussianTh/2
constexpr int gaussianSize = 8;

constexpr int gaussianTileX = 2;
constexpr int gaussianTileY = 2;

constexpr int tamponWidth = (gaussianThx*gaussianTileX + 2*gaussianSize);
constexpr int tamponHeight = (gaussianThy*gaussianTileY + 2*gaussianSize);

//gaussianSize is 8 which means 0.1% error for sigma 3 (exp(-64/9))
class GaussianHandle{
public:
    float* gaussiankernel_d = NULL;
    float* gaussiankernel_integral_d = NULL;
    void init(){
        const float sigma = pu_dilate;
        float gaussiankernel[4*gaussianSize+3];
        gaussiankernel[2*gaussianSize+1] = 0; //will be integral
        for (int i = 0; i < 2*gaussianSize+1; i++){
            gaussiankernel[i] = std::exp(-(gaussianSize-i)*(gaussianSize-i)/(2*sigma*sigma))/(std::sqrt(TAU*sigma*sigma));
            gaussiankernel[2*gaussianSize+2+i] = gaussiankernel[2*gaussianSize+1+i] + gaussiankernel[i];
        }
    
        GPU_CHECK(hipMalloc(&gaussiankernel_d, sizeof(float)*(4*gaussianSize+3)));
        GPU_CHECK(hipMemcpyHtoD((hipDeviceptr_t)gaussiankernel_d, gaussiankernel, (4*gaussianSize+3)*sizeof(float)));
        gaussiankernel_integral_d = gaussiankernel_d + 2*gaussianSize+1;
    }
    void destroy(){
        GPU_CHECK(hipFree(gaussiankernel_d));
    }
};

__device__ void GaussianSmartSharedLoadMinAbs(float* tampon, const float* src1, const float* src2, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - gaussianSize;
    const int tampon_base_y = y - thy - gaussianSize;
    const int thid = thy*gaussianThx+thx;

    #pragma unroll
    for (int i = thid; i < tamponWidth*tamponHeight; i+=gaussianThx*gaussianThy){
        const int tamponX = i%tamponWidth;
        const int tamponY = i/tamponWidth;

        tampon[i] = (tampon_base_x + tamponX >= 0 && tampon_base_x + tamponX < width && tampon_base_y + tamponY >= 0 && tampon_base_y + tamponY < height) ? min(abs(src1[(tampon_base_y+tamponY)*width + tampon_base_x+tamponX]), abs(src2[(tampon_base_y+tamponY)*width + tampon_base_x+tamponX])) : 0.f;
    }
    __syncthreads();
}

__device__ void GaussianSmartSharedLoad(float* tampon, const float* src, int64_t x, int64_t y, int64_t width, int64_t height){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;
    const int tampon_base_x = x - thx - gaussianSize;
    const int tampon_base_y = y - thy - gaussianSize;
    const int thid = thy*gaussianThx+thx;

    #pragma unroll
    for (int i = thid; i < tamponWidth*tamponHeight; i+=gaussianThx*gaussianThy){
        const int tamponX = i%tamponWidth;
        const int tamponY = i/tamponWidth;

        tampon[i] = (tampon_base_x + tamponX >= 0 && tampon_base_x + tamponX < width && tampon_base_y + tamponY >= 0 && tampon_base_y + tamponY < height) ? src[(tampon_base_y+tamponY)*width + tampon_base_x+tamponX] : 0.f;
    }
    __syncthreads();
}

//the following comment is only true if you uncomment putting the end result in the tampon, currently, the result is only put in output
//at the end, the central (tamponWidth-gaussianSize*2)*(tamponHeight-gaussianSize*2) part of tampon contains the blurred value for each thread
__device__ void GaussianSmart_Device(float output[gaussianTileX*gaussianTileY], float* tampon, int64_t x, int64_t y, int64_t width, int64_t height, float* gaussiankernel, float* gaussiankernel_integral){
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;

    float tot[gaussianTileX];
    float out[gaussianTileX][gaussianTileY+1];
    for (int i = 0; i < gaussianTileX; i++) for (int j = 0; j < gaussianTileY+1; j++) out[i][j] = 0.f;

    //horizontal blur on each X tile columns (each of size gaussianTileX * tamponHeight)
    #pragma unroll
    for (int region_x = 0; region_x < gaussianTileX; region_x++){

        //border handling precompute
        const int beg = max((int64_t)0, x-gaussianSize+region_x*gaussianThx)-(x-gaussianSize+region_x*gaussianThx);
        const int end2 = min(width, x+gaussianSize+1+region_x*gaussianThx)-min(width, (x-gaussianSize+region_x*gaussianThx));
        tot[region_x] = gaussiankernel_integral[end2]-gaussiankernel_integral[beg];

        //if (2*gaussianSize)%gaussianThy != 0, we will have idle threads at last step
        int tileRow = 0;
        #pragma unroll
        for (int tamponY = thy; tamponY < tamponHeight; tamponY += gaussianThy){
            for (int i = 0; i < 2*gaussianSize+1; i++){ //starting gaussianSize to the left and going gaussianSize to the right
                out[region_x][tileRow] += tampon[tamponY*tamponWidth + thx+i+region_x*gaussianThx]*gaussiankernel[i];
            }
            tileRow++;
        }
    }

    //rewrite result in tampon and normalize
    __syncthreads();
    #pragma unroll
    for (int region_x = 0; region_x < gaussianTileX; region_x++){
        int tileRow = 0;
        #pragma unroll
        for (int tamponY = thy; tamponY < tamponHeight; tamponY += gaussianThy){
            tampon[(tamponY)*tamponWidth + thx+gaussianSize+region_x*gaussianThx] = out[region_x][tileRow]/tot[region_x];
            tileRow++;
        }
    }
    __syncthreads();

    //verticalBlur on tampon restraint into final center rectangle.
    #pragma unroll
    for (int region_y = 0; region_y < gaussianTileY; region_y++){

        //border handling precompute
        const int beg = max((int64_t)0, y-gaussianSize+region_y*gaussianThy)-(y-gaussianSize+region_y*gaussianThy);
        const int end2 = min(height, y+gaussianSize+1+region_y*gaussianThy)-min(height, (y-gaussianSize+region_y*gaussianThy));
        tot[region_y] = gaussiankernel_integral[end2] - gaussiankernel_integral[beg];

        #pragma unroll
        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            out[region_y][region_x] = 0;
            for (int i = 0; i < 2*gaussianSize+1; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*gaussianThy)*tamponWidth + thx+gaussianSize+region_x*gaussianThx]*gaussiankernel[i];
            }
        }
    }

    //put data to result, one data point per Tile
    #pragma unroll
    for (int region_y = 0; region_y < gaussianTileY; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            output[region_y*gaussianTileX+region_x] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();
    //put data back to tampon -> extra operations and sync
    /*
    #pragma unroll
    for (int region_y = 0; region_y < gaussianTileY; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < gaussianTileX; region_x++){
            tampon[(thy+gaussianSize+region_y*gaussianThy)*tamponWidth + thx+gaussianSize+region_x*gaussianThx] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();
    */
}

}
