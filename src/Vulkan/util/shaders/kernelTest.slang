[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID,uniform float* data){
    uint index = threadId.x;
    data[index] = data[index] + 1;
}