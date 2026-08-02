__device__ float mircuda_bf16_to_float(unsigned short value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16u);
}

__device__ unsigned short mircuda_float_to_bf16(float value) {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16u) & 1u);
  return static_cast<unsigned short>(bits >> 16u);
}

__device__ float mircuda_e4m3(unsigned char value) {
  unsigned int magnitude = value & 0x7fu;
  unsigned int exponent = magnitude >> 3u;
  unsigned int mantissa = magnitude & 7u;
  if (exponent == 15u && mantissa == 7u) {
    return __int_as_float(0x7fc00000);
  }
  float decoded = exponent == 0u
      ? ldexpf(static_cast<float>(mantissa), -9)
      : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
               static_cast<int>(exponent) - 7);
  return (value & 0x80u) == 0u ? decoded : -decoded;
}

__device__ float mircuda_e8m0(unsigned char value) {
  return value == 255u
      ? __int_as_float(0x7fc00000)
      : ldexpf(1.0f, static_cast<int>(value) - 127);
}

__device__ unsigned char mircuda_encode_e4m3(float value) {
  unsigned char sign = (__float_as_uint(value) >> 31u) == 0u ? 0u : 0x80u;
  float magnitude = fabsf(value);
  if (magnitude == 0.0f) return sign;
  if (magnitude >= 448.0f) return sign | 0x7eu;
  if (magnitude < 0.015625f) {
    unsigned int mantissa = static_cast<unsigned int>(nearbyintf(magnitude * 512.0f));
    return sign | static_cast<unsigned char>(min(8u, mantissa));
  }
  int exponent = ilogbf(magnitude);
  unsigned int encoded_exponent = static_cast<unsigned int>(exponent + 7);
  unsigned int mantissa = static_cast<unsigned int>(
      nearbyintf((ldexpf(magnitude, -exponent) - 1.0f) * 8.0f));
  if (mantissa == 8u) {
    ++encoded_exponent;
    mantissa = 0u;
  }
  if (encoded_exponent >= 15u) {
    encoded_exponent = 15u;
    mantissa = min(6u, mantissa);
  }
  return sign | static_cast<unsigned char>((encoded_exponent << 3u) | mantissa);
}

__device__ unsigned int mircuda_mxfp8_scale_offset(
    unsigned int row, unsigned int column,
    unsigned int rows, unsigned int columns) {
  unsigned int column_tiles = (columns + 3u) / 4u;
  unsigned int row_tile = row / 128u;
  unsigned int row_in_tile = row % 128u;
  unsigned int row_quarter = row_in_tile / 32u;
  unsigned int row_lane = row_in_tile % 32u;
  unsigned int column_tile = column / 4u;
  unsigned int column_in_tile = column % 4u;
  return ((((row_tile * column_tiles + column_tile) * 32u + row_lane) * 4u +
            row_quarter) * 4u + column_in_tile);
}

extern "C" __global__ void mircuda_mxfp8_quantize_bf16(
    const unsigned short* input,
    unsigned char* output,
    unsigned char* scales,
    unsigned int rows,
    unsigned int columns) {
  unsigned int group = blockIdx.x;
  unsigned int row = blockIdx.y;
  unsigned int lane = threadIdx.x;
  unsigned int column = group * 32u + lane;
  if (row >= rows || column >= columns) return;
  float value = mircuda_bf16_to_float(input[row * columns + column]);
  float maximum = fabsf(value);
  for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
  }
  maximum = __shfl_sync(0xffffffffu, maximum, 0u);
  float requested_scale = fmaxf(maximum / 448.0f, 5.87747175e-39f);
  int exponent = max(0, min(254,
      static_cast<int>(ceilf(log2f(requested_scale))) + 127));
  float scale = ldexpf(1.0f, exponent - 127);
  output[row * columns + column] = mircuda_encode_e4m3(value / scale);
  if (lane == 0u) {
    unsigned int groups = columns / 32u;
    scales[mircuda_mxfp8_scale_offset(row, group, rows, groups)] =
        static_cast<unsigned char>(exponent);
  }
}

extern "C" __global__ void mircuda_mxfp8_swizzle_scales(
    const unsigned char* input,
    unsigned char* output,
    unsigned int rows,
    unsigned int columns) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int elements = rows * columns;
  if (index >= elements) return;
  unsigned int row = index / columns;
  unsigned int column = index % columns;
  output[mircuda_mxfp8_scale_offset(row, column, rows, columns)] = input[index];
}

extern "C" __global__ void mircuda_mxfp8_bf16(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned char* scales,
    unsigned short* output,
    unsigned int tokens,
    unsigned int input_features,
    unsigned int output_features) {
  unsigned int row = blockIdx.x;
  unsigned int token = blockIdx.y;
  if (row >= output_features || token >= tokens) return;

  unsigned int words = input_features / 4u;
  unsigned int groups = input_features / 32u;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < input_features;
       column += blockDim.x) {
    unsigned int packed = weight[row * words + column / 4u];
    unsigned int shift = (column & 3u) * 8u;
    unsigned char value = static_cast<unsigned char>(packed >> shift);
    float x = mircuda_bf16_to_float(input[token * input_features + column]);
    float scale = mircuda_e8m0(scales[row * groups + column / 32u]);
    sum += x * mircuda_e4m3(value) * scale;
  }

  unsigned int lane = threadIdx.x & 31u;
  unsigned int warp = threadIdx.x >> 5u;
  for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warp_sums[8];
  if (lane == 0u) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0u) {
    sum = lane < 8u ? warp_sums[lane] : 0.0f;
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
      output[token * output_features + row] = mircuda_float_to_bf16(sum);
    }
  }
}

extern "C" __global__ void mircuda_mxfp8_bf16_tiled(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned char* scales,
    unsigned short* output,
    unsigned int tokens,
    unsigned int input_features,
    unsigned int output_features) {
  constexpr unsigned int kTokenTile = 4u;
  unsigned int row = blockIdx.x;
  unsigned int token_start = blockIdx.y * kTokenTile;
  if (row >= output_features || token_start >= tokens) return;

  unsigned int words = input_features / 4u;
  unsigned int groups = input_features / 32u;
  float sums[kTokenTile] = {};
  for (unsigned int column = threadIdx.x; column < input_features;
       column += blockDim.x) {
    unsigned int packed = weight[row * words + column / 4u];
    unsigned int shift = (column & 3u) * 8u;
    unsigned char value = static_cast<unsigned char>(packed >> shift);
    float decoded_weight = mircuda_e4m3(value);
    float scale = mircuda_e8m0(scales[row * groups + column / 32u]);
    #pragma unroll
    for (unsigned int item = 0; item < kTokenTile; ++item) {
      unsigned int token = token_start + item;
      if (token < tokens) {
        float x = mircuda_bf16_to_float(input[token * input_features + column]);
        sums[item] += x * decoded_weight * scale;
      }
    }
  }

  unsigned int lane = threadIdx.x & 31u;
  unsigned int warp = threadIdx.x >> 5u;
  #pragma unroll
  for (unsigned int item = 0; item < kTokenTile; ++item) {
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
      sums[item] += __shfl_down_sync(0xffffffffu, sums[item], offset);
    }
  }
  __shared__ float warp_sums[kTokenTile][8];
  if (lane == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < kTokenTile; ++item) {
      warp_sums[item][warp] = sums[item];
    }
  }
  __syncthreads();
  if (warp == 0u) {
    #pragma unroll
    for (unsigned int item = 0; item < kTokenTile; ++item) {
      float sum = lane < 8u ? warp_sums[item][lane] : 0.0f;
      for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      }
      unsigned int token = token_start + item;
      if (lane == 0u && token < tokens) {
        output[token * output_features + row] = mircuda_float_to_bf16(sum);
      }
    }
  }
}

extern "C" __global__ void mircuda_mxfp8_bf16_gathered(
    const unsigned short* input,
    const unsigned int* weight,
    const unsigned char* scales,
    const unsigned short* bias,
    const unsigned int* selected,
    unsigned short* output,
    unsigned int assignments,
    unsigned int matrices,
    unsigned int rows,
    unsigned int columns,
    unsigned int selections_per_input,
    unsigned int has_bias) {
  unsigned int row = blockIdx.x;
  unsigned int assignment = blockIdx.y;
  if (row >= rows || assignment >= assignments) return;
  unsigned int matrix = selected[assignment];
  if (matrix >= matrices) {
    if (threadIdx.x == 0u) output[assignment * rows + row] = mircuda_float_to_bf16(0.0f);
    return;
  }
  unsigned int words = columns / 4u;
  unsigned int groups = columns / 32u;
  unsigned int matrix_row = matrix * rows + row;
  unsigned int input_row = assignment / selections_per_input;
  float sum = 0.0f;
  for (unsigned int column = threadIdx.x; column < columns; column += blockDim.x) {
    unsigned int packed = weight[matrix_row * words + column / 4u];
    unsigned char value = static_cast<unsigned char>(packed >> ((column & 3u) * 8u));
    float x = mircuda_bf16_to_float(input[input_row * columns + column]);
    float scale = mircuda_e8m0(scales[matrix_row * groups + column / 32u]);
    sum += x * mircuda_e4m3(value) * scale;
  }
  unsigned int lane = threadIdx.x & 31u;
  unsigned int warp = threadIdx.x >> 5u;
  for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  __shared__ float warp_sums[8];
  if (lane == 0u) warp_sums[warp] = sum;
  __syncthreads();
  if (warp == 0u) {
    unsigned int warps = blockDim.x >> 5u;
    sum = lane < warps ? warp_sums[lane] : 0.0f;
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0u) {
      float value = sum + (has_bias == 0u ? 0.0f : mircuda_bf16_to_float(bias[matrix_row]));
      output[assignment * rows + row] = mircuda_float_to_bf16(value);
    }
  }
}

extern "C" __global__ void mircuda_mxfp8_embedding_bf16(
    const unsigned int* weight,
    const unsigned char* scales,
    const unsigned int* selected,
    unsigned short* output,
    unsigned int selected_start,
    unsigned int tokens,
    unsigned int vocab,
    unsigned int hidden,
    float output_scale) {
  unsigned int token = blockIdx.x;
  if (token >= tokens) return;
  unsigned int row = selected[selected_start + token];
  if (row >= vocab) return;
  unsigned int words = hidden / 4u;
  unsigned int groups = hidden / 32u;
  for (unsigned int column = threadIdx.x; column < hidden; column += blockDim.x) {
    unsigned int packed = weight[row * words + column / 4u];
    unsigned char value = static_cast<unsigned char>(packed >> ((column & 3u) * 8u));
    float scale = mircuda_e8m0(scales[row * groups + column / 32u]);
    output[token * hidden + column] =
        mircuda_float_to_bf16(mircuda_e4m3(value) * scale * output_scale);
  }
}
