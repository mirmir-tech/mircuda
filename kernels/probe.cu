extern "C" __global__ void mircuda_probe(float* output, float value, unsigned int length) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        output[index] = value;
    }
}
