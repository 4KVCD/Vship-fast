static const float jod_a = 0.0439569391310215f;
static const float jod_exp = 0.9302042722702026f;

float toJOD(float a){
    if (a > 0.1f){
        return 10.f - jod_a * pow(a, jod_exp);
    } else {
        const float jod_a_p = jod_a * (pow(0.1f, jod_exp-1.f));
        return 10.f - jod_a_p * a;
    }
}

float topow4(float el){
    float res = el;
    res *= res;
    res *= res;
    return res;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mergeChromaNorm(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, float* p0, uniform float* p1, uniform float* p2, uniform float* p3, uniform int64_t size, uniform float divisor){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= size) return;

    p0[thid] = pow((topow4(p0[thid]) + topow4(p1[thid]) + topow4(p2[thid]) + topow4(p3[thid]))*0.25, 0.25f)/(float)divisor;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void JODIZE(uint3 threadIdx : SV_GroupThreadID, uint3 blockIdx : SV_GroupID, uniform float* p, uniform int64_t size){
    const int64_t thid = threadIdx.x + blockIdx.x * 256;
    if (thid >= size) return;

    p[thid] = 1.f - (toJOD(p[thid]))*0.1f;
}