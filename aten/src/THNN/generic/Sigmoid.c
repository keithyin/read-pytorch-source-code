#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Sigmoid.c"
#else


// THNN 是由宏实现的 一个 类 #define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME) 在 aten/src/THNN/THNN.h 中
// Sigmoid_updateOutput 为啥用 updateOutput 这个 词 呢？
// 这个编译的时候会变成  
// void THNN_RealSigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output);
// 这个才是函数的真面目！！！！！！！！！！！！！！！！
// 作用，给定 input ，求 output
void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  // #define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME), 
  // 这个在编译的时候，会变成 THRealTensor_sigmoid(output, input);
  // THNN_ 比 THTensor 是要多一个 THNNState 
  THTensor_(sigmoid)(output, input);
}

// 作用， 给定 output, gradOutput, 求 gradInput
void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    real z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z;
  );
}

#endif

/*
void THNN_(Sigmoid_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  // #define THTensor_(NAME) TH_CONCAT_4(TH, Real, Tensor_, NAME), 
  // 这个在编译的时候，会变成 THRealTensor_sigmoid(output, input);
  // THNN_ 比 THTensor 是要多一个 THNNState 
  THTensor_(sigmoid)(output, input);
}

被 预处理器操作完后应该是这样的
void THNN_RealSigmoid_updateOutput(THNNState *state, THTensor *input, THTensor *output){
  THRealTensor_sigmoid(output, input);
}
------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
void THNN_(Sigmoid_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_NELEMENT(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);
  
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
    real z = *output_data;
    *gradInput_data = *gradOutput_data * (1. - z) * z;
  );
}
被预处理器操作完后，是这样的
void THNN_RealSigmoid_updateGradInput(THNNState *state, THTensor *gradOutput, THTensor *gradInput, THTensor *output){
  // 先检查一下 output 和 gradOutput 的 元素是否匹配
  THRealTensor_resizeAs(gradInput, output); // 为啥要这样操作, 因为是 sigmoid 嘛， input 是要和 output 一样的咯！！！！！
  

}

#define TH_TENSOR_APPLY3_D(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, -1, CODE) \
{ \
  int TH_TENSOR_APPLY_hasFinished = 0; \
  int64_t TH_TENSOR_dim_index = 0; \
  __TH_TENSOR_APPLYX_PREAMBLE(real, gradInput, -1, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(real, gradOutput, -1, 1) \
  __TH_TENSOR_APPLYX_PREAMBLE(real, output, -1, 1) \
  // 经过这里之后 ，gradInput 变成 gradInput_data ........ 
                                                                        \
  int elements_equal = 1;                                               \
  if(gradInput_n != gradOutput_n) {                                      \
    elements_equal = 0;                                                 \
  }                                                                     \
  else if(gradInput_n != output_n) {                                 \
    elements_equal = 0;                                                 \
  }                                                                     \
  if (elements_equal == 0) {                                            \
    THDescBuff T1buff = _THSizeDesc(gradInput->size, gradInput->nDimension); \
    THDescBuff T2buff = _THSizeDesc(TENSOR2->size, TENSOR2->nDimension); \
    THDescBuff T3buff = _THSizeDesc(TENSOR3->size, TENSOR3->nDimension); \
    THError("inconsistent tensor size, expected %s %s, %s %s and %s %s to have the same " \
            "number of elements, but got %d, %d and %d elements respectively", \
            #TENSOR1, T1buff.str, #TENSOR2, T2buff.str, #TENSOR3, T3buff.str, \
            TENSOR1##_n, TENSOR2##_n, TENSOR3##_n);                     \
  }                                                                     \
                                                                        \
  while(!TH_TENSOR_APPLY_hasFinished) \
  { \
    // Loop through the inner most region of the Tensor  \ gradInput_size 元素个数。
    for(; gradInput_i < gradInput_size && TENSOR2##_i < TENSOR2##_size && TENSOR3##_i < TENSOR3##_size;\
     gradInput_i++, TENSOR2##_i++, TENSOR3##_i++, gradInput_data += gradInput_stride, TENSOR2##_data += TENSOR2##_stride, TENSOR3##_data += TENSOR3##_stride) \
    { \
      CODE \
    } \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR1, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR2, 0) \
    __TH_TENSOR_APPLYX_UPDATE_COUNTERS(TENSOR3, 0) \
  } \
  if(TENSOR1##_counter != NULL) \
    THFree(TENSOR1##_counter); \
  if(TENSOR2##_counter != NULL) \
    THFree(TENSOR2##_counter); \
  if(TENSOR3##_counter != NULL) \
    THFree(TENSOR3##_counter); \
}
*/


// 下面代码的一些注释
/*
  TENSOR_i 表示 Tensor 的 维度！！！！！！！！！！！！！！！！
  TENSOR_n 用来保存 Tensor中 元素的个数。
  TENSOR_data 表示数据的起始 地址
  TENSOR_size : 
*/

/*

#define __TH_TENSOR_APPLYX_PREAMBLE(TYPE, TENSOR, DIM, ALLOW_CONTIGUOUS) 
  TYPE *TENSOR##_data = NULL; 
  int64_t *TENSOR##_counter = NULL, *TENSOR##_sizes = NULL, *TENSOR##_strides = NULL, *TENSOR##_dimOffset = NULL; \
  int64_t TENSOR##_stride = 0, TENSOR##_size = 0, TENSOR##_dim = 0, TENSOR##_i, TENSOR##_n; \
  int TENSOR##_contiguous = ALLOW_CONTIGUOUS && DIM < 0; \
  TENSOR##_n = (TENSOR->nDimension ? 1 : 0); \
  // TENSOR_i 表示 Tensor 的 维度！！！！！！！！！！！！！！！！
  // TENSOR_n 表示 tensor->size[i]
  for(TENSOR##_i = 0; TENSOR##_i < TENSOR->nDimension; TENSOR##_i++) \
    TENSOR##_n *= TENSOR->size[TENSOR##_i]; \
\
  if(TENSOR->nDimension == 0) \
    TH_TENSOR_APPLY_hasFinished = 1; \
  else \
  { \
    TENSOR##_data = TENSOR->storage->data+TENSOR->storageOffset; // 起使指针 加偏移。
    TENSOR##_size = 1; \
    TENSOR##_stride = 1; \
    // 从 OUTERmost 开始 循环！！！！！！！！！！！！！！！
    for(TENSOR##_i = TENSOR->nDimension-1; TENSOR##_i >= 0; TENSOR##_i--) { \
      if(TENSOR->size[TENSOR##_i] != 1) { 


        // 这地方是干嘛的？？？？？？？？？？？？？？？？
        if(TENSOR->stride[TENSOR##_i] == TENSOR##_size && TENSOR##_i != DIM) \
          TENSOR##_size *= TENSOR->size[TENSOR##_i]; \
        else{ \
          TENSOR##_contiguous = 0; \
          break; \
        } \
      } \
    } \

    // 如果 Tensor 不连续，要 搞什么呢？
    if (!TENSOR##_contiguous) { \
      // Find the dimension of contiguous sections  
      TENSOR##_dim = 1; \
      for(TENSOR##_i = TENSOR->nDimension-2; TENSOR##_i >= 0; TENSOR##_i--) \
      { \
        if(TENSOR->stride[TENSOR##_i] != TENSOR->stride[TENSOR##_i+1] * TENSOR->size[TENSOR##_i+1] || TENSOR##_i == DIM || TENSOR##_i+1 == DIM) \
          TENSOR##_dim++; \
      } \
      // Allocate an array of 3*dim elements, where dim is the number of contiguous sections 

      TENSOR##_counter = (int64_t*)THAlloc(sizeof(int64_t)*(3*TENSOR##_dim)); \
      TENSOR##_sizes = TENSOR##_counter + TENSOR##_dim; \ 一部分 放 sizes， 一部分放 strides
      TENSOR##_strides = TENSOR##_counter + 2*TENSOR##_dim; \
      TH_TENSOR_dim_index = TENSOR##_dim-1; \
      TENSOR##_dimOffset = (DIM == TENSOR->nDimension-1) ? &TENSOR##_i : &TENSOR##_counter[DIM]; \
      TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR->nDimension-1]; \
      TENSOR##_strides[TH_TENSOR_dim_index] = TENSOR->stride[TENSOR->nDimension-1]; \
      // TENSOR##_counter tracks where we are in the storage. The offset into the  \
      // storage is given by storage_offset + (i * j), where i is the stride  \
      // vector and j is tensor_counter vector. This sets the starting position for the loop. 
      for(TENSOR##_i = TENSOR##_dim-1; TENSOR##_i >= 0; --TENSOR##_i) { \
        TENSOR##_counter[TENSOR##_i] = 0; \
      } \
      for(TENSOR##_i = TENSOR->nDimension-2; TENSOR##_i >= 0; --TENSOR##_i) { \
        if (TENSOR->stride[TENSOR##_i] == TENSOR->stride[TENSOR##_i+1] * TENSOR->size[TENSOR##_i+1] && TENSOR##_i != DIM && TENSOR##_i+1 != DIM) { \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR##_i] * TENSOR##_sizes[TH_TENSOR_dim_index]; \
          if (DIM != TENSOR->nDimension-1 && TENSOR##_i < DIM) \
            TENSOR##_dimOffset--; \
        } else { \
          --TH_TENSOR_dim_index; \
          TENSOR##_sizes[TH_TENSOR_dim_index] = TENSOR->size[TENSOR##_i]; \
          TENSOR##_strides[TH_TENSOR_dim_index] = TENSOR->stride[TENSOR##_i]; \
        } \
      } \
      // Size of the inner most section \
      TENSOR##_size = TENSOR##_sizes[TENSOR##_dim-1]; \
      // Stride of the inner most section  \
      TENSOR##_stride = TENSOR##_strides[TENSOR##_dim-1]; \
    } \
  } \
  TENSOR##_i = 0;

*/