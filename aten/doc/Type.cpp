#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Scalar.h"
#include "ATen/SparseTensorRef.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/UndefinedType.h"

#include <iostream>
#include "ATen/CPUByteType.h"
#include "ATen/CPUCharType.h"
#include "ATen/CPUDoubleType.h"
#include "ATen/CPUFloatType.h"
#include "ATen/CPUIntType.h"
#include "ATen/CPULongType.h"
#include "ATen/CPUShortType.h"
#include "ATen/CPUHalfType.h"
#include "ATen/SparseCPUByteType.h"
#include "ATen/SparseCPUCharType.h"
#include "ATen/SparseCPUDoubleType.h"
#include "ATen/SparseCPUFloatType.h"
#include "ATen/SparseCPUIntType.h"
#include "ATen/SparseCPULongType.h"
#include "ATen/SparseCPUShortType.h"
#include "ATen/CUDAByteType.h"
#include "ATen/CUDACharType.h"
#include "ATen/CUDADoubleType.h"
#include "ATen/CUDAFloatType.h"
#include "ATen/CUDAIntType.h"
#include "ATen/CUDALongType.h"
#include "ATen/CUDAShortType.h"
#include "ATen/CUDAHalfType.h"
#include "ATen/SparseCUDAByteType.h"
#include "ATen/SparseCUDACharType.h"
#include "ATen/SparseCUDADoubleType.h"
#include "ATen/SparseCUDAFloatType.h"
#include "ATen/SparseCUDAIntType.h"
#include "ATen/SparseCUDALongType.h"
#include "ATen/SparseCUDAShortType.h"

namespace at {

void Type::registerAll(Context * context) {
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Byte)].reset(new CPUByteType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Char)].reset(new CPUCharType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Double)].reset(new CPUDoubleType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Float)].reset(new CPUFloatType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Int)].reset(new CPUIntType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Long)].reset(new CPULongType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Short)].reset(new CPUShortType(context));
  context->type_registry[static_cast<int>(Backend::CPU)][static_cast<int>(ScalarType::Half)].reset(new CPUHalfType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Byte)].reset(new SparseCPUByteType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Char)].reset(new SparseCPUCharType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Double)].reset(new SparseCPUDoubleType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Float)].reset(new SparseCPUFloatType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Int)].reset(new SparseCPUIntType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Long)].reset(new SparseCPULongType(context));
  context->type_registry[static_cast<int>(Backend::SparseCPU)][static_cast<int>(ScalarType::Short)].reset(new SparseCPUShortType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Byte)].reset(new CUDAByteType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Char)].reset(new CUDACharType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Double)].reset(new CUDADoubleType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Float)].reset(new CUDAFloatType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Int)].reset(new CUDAIntType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Long)].reset(new CUDALongType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Short)].reset(new CUDAShortType(context));
  context->type_registry[static_cast<int>(Backend::CUDA)][static_cast<int>(ScalarType::Half)].reset(new CUDAHalfType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Byte)].reset(new SparseCUDAByteType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Char)].reset(new SparseCUDACharType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Double)].reset(new SparseCUDADoubleType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Float)].reset(new SparseCUDAFloatType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Int)].reset(new SparseCUDAIntType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Long)].reset(new SparseCUDALongType(context));
  context->type_registry[static_cast<int>(Backend::SparseCUDA)][static_cast<int>(ScalarType::Short)].reset(new SparseCUDAShortType(context));
  context->type_registry[static_cast<int>(Backend::Undefined)][static_cast<int>(ScalarType::Undefined)].reset(new UndefinedType(context));
}

void Type::copy(const Tensor & src, Tensor & dst) const {
  Tensor b_src;
  std::tie(b_src) = expand_inplace(dst, src, "copy");
  s_copy(b_src, dst);
}

Tensor Type::copy(const Tensor & src) const {
  AT_ASSERT(src.defined(), "attempt to copy an undefined tensor");
  Tensor r = this->tensor(src.sizes());
  r.copy_(src);
  return r;
}

Type & Type::toBackend(Backend b) const {
  return context->getType(b,scalarType());
}
Type & Type::toScalarType(ScalarType s) const {
  return context->getType(backend(),s);
}

Tensor Type::tensorFromBlob(void * data, IntList sizes, const std::function<void(void*)> & deleter) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for(size_t i = sizes.size(); i > 0; --i) {
    strides[i-1] = stride;
    stride *= sizes[i-1];
  }
  return tensorFromBlob(data, sizes, strides, deleter);
}
Tensor Type::tensorFromBlob(void * data, IntList sizes, IntList strides, const std::function<void(void*)> & deleter) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for(size_t i = 0; i < sizes.size(); i++) {
    if(sizes[i] == 0) {
      size = 0;
      break;
    }
    size += strides[i]*(sizes[i]-1);
  }
  auto storage = storageFromBlob(data,size,deleter);
  return tensor(*storage, 0, sizes, strides);
}
Tensor Type::scalarTensor(Scalar s) const {
  if(s.isBackedByTensor())
    return Tensor(s.t).toType(*this);
  return tensor({}).fill_(s);
}

bool Type::operator==(const Type& other) const {
  return this == &other;
}

int64_t Type::storage_offset(const Tensor & self) const {
    runtime_error("storage_offset is not implemented for type %s", toString());
}
Tensor & Type::resize_(Tensor & self, IntList size) const {
    runtime_error("resize_ is not implemented for type %s", toString());
}
Tensor & Type::zeros_out(Tensor & result, IntList size) const {
    runtime_error("zeros_out is not implemented for type %s", toString());
}
Tensor Type::zeros(IntList size) const {
    runtime_error("zeros is not implemented for type %s", toString());
}
Tensor & Type::zeros_like_out(Tensor & result, const Tensor & input) const {
    runtime_error("zeros_like_out is not implemented for type %s", toString());
}
Tensor Type::zeros_like(const Tensor & input) const {
    runtime_error("zeros_like is not implemented for type %s", toString());
}
Tensor & Type::ones_out(Tensor & result, IntList size) const {
    runtime_error("ones_out is not implemented for type %s", toString());
}
Tensor Type::ones(IntList size) const {
    runtime_error("ones is not implemented for type %s", toString());
}
Tensor & Type::ones_like_out(Tensor & result, const Tensor & input) const {
    runtime_error("ones_like_out is not implemented for type %s", toString());
}
Tensor Type::ones_like(const Tensor & input) const {
    runtime_error("ones_like is not implemented for type %s", toString());
}
int64_t Type::numel(const Tensor & self) const {
    runtime_error("numel is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, Storage & storage) const {
    runtime_error("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, Storage & sourceStorage, int64_t storage_offset, IntList size, IntList stride) const {
    runtime_error("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self, const Tensor & source) const {
    runtime_error("set_ is not implemented for type %s", toString());
}
Tensor & Type::set_(Tensor & self) const {
    runtime_error("set_ is not implemented for type %s", toString());
}
Tensor & Type::fill_(Tensor & self, Scalar value) const {
    runtime_error("fill_ is not implemented for type %s", toString());
}
Tensor & Type::fill_(Tensor & self, const Tensor & value) const {
    runtime_error("fill_ is not implemented for type %s", toString());
}
bool Type::is_contiguous(const Tensor & self) const {
    runtime_error("is_contiguous is not implemented for type %s", toString());
}
bool Type::is_set_to(const Tensor & self, const Tensor & tensor) const {
    runtime_error("is_set_to is not implemented for type %s", toString());
}
Tensor & Type::s_masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    runtime_error("s_masked_fill_ is not implemented for type %s", toString());
}
Tensor & Type::masked_fill_(Tensor & self, const Tensor & mask, Scalar value) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_fill_");
    return s_masked_fill_(self, b_mask, value);
}
Tensor & Type::s_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    runtime_error("s_masked_fill_ is not implemented for type %s", toString());
}
Tensor & Type::masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_fill_");
    return s_masked_fill_(self, b_mask, value);
}
Tensor & Type::s_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    runtime_error("s_masked_scatter_ is not implemented for type %s", toString());
}
Tensor & Type::masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source) const {
    Tensor b_mask;
    std::tie(b_mask) = expand_inplace(self, mask, "masked_scatter_");
    return s_masked_scatter_(self, b_mask, source);
}
Tensor & Type::s_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    runtime_error("s_masked_select_out is not implemented for type %s", toString());
}
Tensor & Type::masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) const {
    Tensor b_self, b_mask;
    std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select_out");
    return s_masked_select_out(result, b_self, b_mask);
}
Tensor Type::s_masked_select(const Tensor & self, const Tensor & mask) const {
    runtime_error("s_masked_select is not implemented for type %s", toString());
}
Tensor Type::masked_select(const Tensor & self, const Tensor & mask) const {
    Tensor b_self, b_mask;
    std::tie(b_self, b_mask) = expand_outplace(self, mask, "masked_select");
    return s_masked_select(b_self, b_mask);
}
Tensor Type::transpose(const Tensor & self, int64_t dim0, int64_t dim1) const {
    runtime_error("transpose is not implemented for type %s", toString());
}
Tensor & Type::transpose_(Tensor & self, int64_t dim0, int64_t dim1) const {
    runtime_error("transpose_ is not implemented for type %s", toString());
}
Tensor Type::t(const Tensor & self) const {
    runtime_error("t is not implemented for type %s", toString());
}
Tensor & Type::t_(Tensor & self) const {
    runtime_error("t_ is not implemented for type %s", toString());
}
Tensor & Type::nonzero_out(Tensor & result, const Tensor & self) const {
    runtime_error("nonzero_out is not implemented for type %s", toString());
}
Tensor Type::nonzero(const Tensor & self) const {
    runtime_error("nonzero is not implemented for type %s", toString());
}
Tensor Type::contiguous(const Tensor & self) const {
    runtime_error("contiguous is not implemented for type %s", toString());
}
Tensor Type::clone(const Tensor & self) const {
    runtime_error("clone is not implemented for type %s", toString());
}
Tensor Type::view(const Tensor & self, IntList size) const {
    runtime_error("view is not implemented for type %s", toString());
}
Tensor & Type::resize_as_(Tensor & self, const Tensor & the_template) const {
    runtime_error("resize_as_ is not implemented for type %s", toString());
}
Tensor & Type::index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    runtime_error("index_select_out is not implemented for type %s", toString());
}
Tensor Type::index_select(const Tensor & self, int64_t dim, const Tensor & index) const {
    runtime_error("index_select is not implemented for type %s", toString());
}
Tensor & Type::index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    runtime_error("index_copy_ is not implemented for type %s", toString());
}
Tensor & Type::take_out(Tensor & result, const Tensor & self, const Tensor & index) const {
    runtime_error("take_out is not implemented for type %s", toString());
}
Tensor Type::take(const Tensor & self, const Tensor & index) const {
    runtime_error("take is not implemented for type %s", toString());
}
Tensor & Type::put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate) const {
    runtime_error("put_ is not implemented for type %s", toString());
}
Tensor & Type::index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) const {
    runtime_error("index_add_ is not implemented for type %s", toString());
}
Tensor & Type::index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    runtime_error("index_fill_ is not implemented for type %s", toString());
}
Tensor & Type::index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value) const {
    runtime_error("index_fill_ is not implemented for type %s", toString());
}
Tensor Type::unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) const {
    runtime_error("unfold is not implemented for type %s", toString());
}
Tensor & Type::range_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    runtime_error("range_out is not implemented for type %s", toString());
}
Tensor Type::range(Scalar start, Scalar end, Scalar step) const {
    runtime_error("range is not implemented for type %s", toString());
}
Tensor & Type::arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) const {
    runtime_error("arange_out is not implemented for type %s", toString());
}
Tensor Type::arange(Scalar start, Scalar end, Scalar step) const {
    runtime_error("arange is not implemented for type %s", toString());
}
Tensor & Type::arange_out(Tensor & result, Scalar end) const {
    runtime_error("arange_out is not implemented for type %s", toString());
}
Tensor Type::arange(Scalar end) const {
    runtime_error("arange is not implemented for type %s", toString());
}
Tensor & Type::scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    runtime_error("scatter_ is not implemented for type %s", toString());
}
Tensor & Type::scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value) const {
    runtime_error("scatter_ is not implemented for type %s", toString());
}
Tensor & Type::scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src) const {
    runtime_error("scatter_add_ is not implemented for type %s", toString());
}
Tensor & Type::gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) const {
    runtime_error("gather_out is not implemented for type %s", toString());
}
Tensor Type::gather(const Tensor & self, int64_t dim, const Tensor & index) const {
    runtime_error("gather is not implemented for type %s", toString());
}
void* Type::data_ptr(const Tensor & self) const {
    runtime_error("data_ptr is not implemented for type %s", toString());
}
bool Type::equal(const Tensor & self, const Tensor & other) const {
    runtime_error("equal is not implemented for type %s", toString());
}
Tensor & Type::__and___out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("__and___out is not implemented for type %s", toString());
}
Tensor Type::__and__(const Tensor & self, Scalar other) const {
    runtime_error("__and__ is not implemented for type %s", toString());
}
Tensor & Type::s___and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s___and___out is not implemented for type %s", toString());
}
Tensor & Type::__and___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__and___out");
    return s___and___out(result, b_self, b_other);
}
Tensor Type::s___and__(const Tensor & self, const Tensor & other) const {
    runtime_error("s___and__ is not implemented for type %s", toString());
}
Tensor Type::__and__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__and__");
    return s___and__(b_self, b_other);
}
Tensor & Type::__iand__(Tensor & self, Scalar other) const {
    runtime_error("__iand__ is not implemented for type %s", toString());
}
Tensor & Type::s___iand__(Tensor & self, const Tensor & other) const {
    runtime_error("s___iand__ is not implemented for type %s", toString());
}
Tensor & Type::__iand__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__iand__");
    return s___iand__(self, b_other);
}
Tensor & Type::__or___out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("__or___out is not implemented for type %s", toString());
}
Tensor Type::__or__(const Tensor & self, Scalar other) const {
    runtime_error("__or__ is not implemented for type %s", toString());
}
Tensor & Type::s___or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s___or___out is not implemented for type %s", toString());
}
Tensor & Type::__or___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__or___out");
    return s___or___out(result, b_self, b_other);
}
Tensor Type::s___or__(const Tensor & self, const Tensor & other) const {
    runtime_error("s___or__ is not implemented for type %s", toString());
}
Tensor Type::__or__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__or__");
    return s___or__(b_self, b_other);
}
Tensor & Type::__ior__(Tensor & self, Scalar other) const {
    runtime_error("__ior__ is not implemented for type %s", toString());
}
Tensor & Type::s___ior__(Tensor & self, const Tensor & other) const {
    runtime_error("s___ior__ is not implemented for type %s", toString());
}
Tensor & Type::__ior__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ior__");
    return s___ior__(self, b_other);
}
Tensor & Type::__xor___out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("__xor___out is not implemented for type %s", toString());
}
Tensor Type::__xor__(const Tensor & self, Scalar other) const {
    runtime_error("__xor__ is not implemented for type %s", toString());
}
Tensor & Type::s___xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s___xor___out is not implemented for type %s", toString());
}
Tensor & Type::__xor___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__xor___out");
    return s___xor___out(result, b_self, b_other);
}
Tensor Type::s___xor__(const Tensor & self, const Tensor & other) const {
    runtime_error("s___xor__ is not implemented for type %s", toString());
}
Tensor Type::__xor__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__xor__");
    return s___xor__(b_self, b_other);
}
Tensor & Type::__ixor__(Tensor & self, Scalar other) const {
    runtime_error("__ixor__ is not implemented for type %s", toString());
}
Tensor & Type::s___ixor__(Tensor & self, const Tensor & other) const {
    runtime_error("s___ixor__ is not implemented for type %s", toString());
}
Tensor & Type::__ixor__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ixor__");
    return s___ixor__(self, b_other);
}
Tensor & Type::__lshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("__lshift___out is not implemented for type %s", toString());
}
Tensor Type::__lshift__(const Tensor & self, Scalar other) const {
    runtime_error("__lshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s___lshift___out is not implemented for type %s", toString());
}
Tensor & Type::__lshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__lshift___out");
    return s___lshift___out(result, b_self, b_other);
}
Tensor Type::s___lshift__(const Tensor & self, const Tensor & other) const {
    runtime_error("s___lshift__ is not implemented for type %s", toString());
}
Tensor Type::__lshift__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__lshift__");
    return s___lshift__(b_self, b_other);
}
Tensor & Type::__ilshift__(Tensor & self, Scalar other) const {
    runtime_error("__ilshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___ilshift__(Tensor & self, const Tensor & other) const {
    runtime_error("s___ilshift__ is not implemented for type %s", toString());
}
Tensor & Type::__ilshift__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__ilshift__");
    return s___ilshift__(self, b_other);
}
Tensor & Type::__rshift___out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("__rshift___out is not implemented for type %s", toString());
}
Tensor Type::__rshift__(const Tensor & self, Scalar other) const {
    runtime_error("__rshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s___rshift___out is not implemented for type %s", toString());
}
Tensor & Type::__rshift___out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__rshift___out");
    return s___rshift___out(result, b_self, b_other);
}
Tensor Type::s___rshift__(const Tensor & self, const Tensor & other) const {
    runtime_error("s___rshift__ is not implemented for type %s", toString());
}
Tensor Type::__rshift__(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "__rshift__");
    return s___rshift__(b_self, b_other);
}
Tensor & Type::__irshift__(Tensor & self, Scalar other) const {
    runtime_error("__irshift__ is not implemented for type %s", toString());
}
Tensor & Type::s___irshift__(Tensor & self, const Tensor & other) const {
    runtime_error("s___irshift__ is not implemented for type %s", toString());
}
Tensor & Type::__irshift__(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "__irshift__");
    return s___irshift__(self, b_other);
}
Tensor & Type::lt_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("lt_out is not implemented for type %s", toString());
}
Tensor Type::lt(const Tensor & self, Scalar other) const {
    runtime_error("lt is not implemented for type %s", toString());
}
Tensor & Type::s_lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_lt_out is not implemented for type %s", toString());
}
Tensor & Type::lt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "lt_out");
    return s_lt_out(result, b_self, b_other);
}
Tensor Type::s_lt(const Tensor & self, const Tensor & other) const {
    runtime_error("s_lt is not implemented for type %s", toString());
}
Tensor Type::lt(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "lt");
    return s_lt(b_self, b_other);
}
Tensor & Type::lt_(Tensor & self, Scalar other) const {
    runtime_error("lt_ is not implemented for type %s", toString());
}
Tensor & Type::s_lt_(Tensor & self, const Tensor & other) const {
    runtime_error("s_lt_ is not implemented for type %s", toString());
}
Tensor & Type::lt_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "lt_");
    return s_lt_(self, b_other);
}
Tensor & Type::gt_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("gt_out is not implemented for type %s", toString());
}
Tensor Type::gt(const Tensor & self, Scalar other) const {
    runtime_error("gt is not implemented for type %s", toString());
}
Tensor & Type::s_gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_gt_out is not implemented for type %s", toString());
}
Tensor & Type::gt_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "gt_out");
    return s_gt_out(result, b_self, b_other);
}
Tensor Type::s_gt(const Tensor & self, const Tensor & other) const {
    runtime_error("s_gt is not implemented for type %s", toString());
}
Tensor Type::gt(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "gt");
    return s_gt(b_self, b_other);
}
Tensor & Type::gt_(Tensor & self, Scalar other) const {
    runtime_error("gt_ is not implemented for type %s", toString());
}
Tensor & Type::s_gt_(Tensor & self, const Tensor & other) const {
    runtime_error("s_gt_ is not implemented for type %s", toString());
}
Tensor & Type::gt_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "gt_");
    return s_gt_(self, b_other);
}
Tensor & Type::le_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("le_out is not implemented for type %s", toString());
}
Tensor Type::le(const Tensor & self, Scalar other) const {
    runtime_error("le is not implemented for type %s", toString());
}
Tensor & Type::s_le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_le_out is not implemented for type %s", toString());
}
Tensor & Type::le_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "le_out");
    return s_le_out(result, b_self, b_other);
}
Tensor Type::s_le(const Tensor & self, const Tensor & other) const {
    runtime_error("s_le is not implemented for type %s", toString());
}
Tensor Type::le(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "le");
    return s_le(b_self, b_other);
}
Tensor & Type::le_(Tensor & self, Scalar other) const {
    runtime_error("le_ is not implemented for type %s", toString());
}
Tensor & Type::s_le_(Tensor & self, const Tensor & other) const {
    runtime_error("s_le_ is not implemented for type %s", toString());
}
Tensor & Type::le_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "le_");
    return s_le_(self, b_other);
}
Tensor & Type::ge_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("ge_out is not implemented for type %s", toString());
}
Tensor Type::ge(const Tensor & self, Scalar other) const {
    runtime_error("ge is not implemented for type %s", toString());
}
Tensor & Type::s_ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_ge_out is not implemented for type %s", toString());
}
Tensor & Type::ge_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ge_out");
    return s_ge_out(result, b_self, b_other);
}
Tensor Type::s_ge(const Tensor & self, const Tensor & other) const {
    runtime_error("s_ge is not implemented for type %s", toString());
}
Tensor Type::ge(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ge");
    return s_ge(b_self, b_other);
}
Tensor & Type::ge_(Tensor & self, Scalar other) const {
    runtime_error("ge_ is not implemented for type %s", toString());
}
Tensor & Type::s_ge_(Tensor & self, const Tensor & other) const {
    runtime_error("s_ge_ is not implemented for type %s", toString());
}
Tensor & Type::ge_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "ge_");
    return s_ge_(self, b_other);
}
Tensor & Type::eq_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("eq_out is not implemented for type %s", toString());
}
Tensor Type::eq(const Tensor & self, Scalar other) const {
    runtime_error("eq is not implemented for type %s", toString());
}
Tensor & Type::s_eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_eq_out is not implemented for type %s", toString());
}
Tensor & Type::eq_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "eq_out");
    return s_eq_out(result, b_self, b_other);
}
Tensor Type::s_eq(const Tensor & self, const Tensor & other) const {
    runtime_error("s_eq is not implemented for type %s", toString());
}
Tensor Type::eq(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "eq");
    return s_eq(b_self, b_other);
}
Tensor & Type::eq_(Tensor & self, Scalar other) const {
    runtime_error("eq_ is not implemented for type %s", toString());
}
Tensor & Type::s_eq_(Tensor & self, const Tensor & other) const {
    runtime_error("s_eq_ is not implemented for type %s", toString());
}
Tensor & Type::eq_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "eq_");
    return s_eq_(self, b_other);
}
Tensor & Type::ne_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("ne_out is not implemented for type %s", toString());
}
Tensor Type::ne(const Tensor & self, Scalar other) const {
    runtime_error("ne is not implemented for type %s", toString());
}
Tensor & Type::s_ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_ne_out is not implemented for type %s", toString());
}
Tensor & Type::ne_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ne_out");
    return s_ne_out(result, b_self, b_other);
}
Tensor Type::s_ne(const Tensor & self, const Tensor & other) const {
    runtime_error("s_ne is not implemented for type %s", toString());
}
Tensor Type::ne(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "ne");
    return s_ne(b_self, b_other);
}
Tensor & Type::ne_(Tensor & self, Scalar other) const {
    runtime_error("ne_ is not implemented for type %s", toString());
}
Tensor & Type::s_ne_(Tensor & self, const Tensor & other) const {
    runtime_error("s_ne_ is not implemented for type %s", toString());
}
Tensor & Type::ne_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "ne_");
    return s_ne_(self, b_other);
}
std::tuple<Tensor &,Tensor &> Type::min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("min_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::min(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("min is not implemented for type %s", toString());
}
Tensor & Type::s_min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_min_out is not implemented for type %s", toString());
}
Tensor & Type::min_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "min_out");
    return s_min_out(result, b_self, b_other);
}
Tensor Type::s_min(const Tensor & self, const Tensor & other) const {
    runtime_error("s_min is not implemented for type %s", toString());
}
Tensor Type::min(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "min");
    return s_min(b_self, b_other);
}
Tensor Type::min(const Tensor & self) const {
    runtime_error("min is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("max_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("max is not implemented for type %s", toString());
}
Tensor & Type::s_max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_max_out is not implemented for type %s", toString());
}
Tensor & Type::max_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "max_out");
    return s_max_out(result, b_self, b_other);
}
Tensor Type::s_max(const Tensor & self, const Tensor & other) const {
    runtime_error("s_max is not implemented for type %s", toString());
}
Tensor Type::max(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "max");
    return s_max(b_self, b_other);
}
Tensor Type::max(const Tensor & self) const {
    runtime_error("max is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    runtime_error("kthvalue_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) const {
    runtime_error("kthvalue is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("mode_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::mode(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("mode is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("median_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::median(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("median is not implemented for type %s", toString());
}
Tensor Type::median(const Tensor & self) const {
    runtime_error("median is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) const {
    runtime_error("sort_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::sort(const Tensor & self, int64_t dim, bool descending) const {
    runtime_error("sort is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    runtime_error("topk_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) const {
    runtime_error("topk is not implemented for type %s", toString());
}
bool Type::all(const Tensor & self) const {
    runtime_error("all is not implemented for type %s", toString());
}
bool Type::any(const Tensor & self) const {
    runtime_error("any is not implemented for type %s", toString());
}
int64_t Type::get_device(const Tensor & self) const {
    runtime_error("get_device is not implemented for type %s", toString());
}
Tensor & Type::abs_out(Tensor & destination, const Tensor & self) const {
    runtime_error("abs_out is not implemented for type %s", toString());
}
Tensor Type::abs(const Tensor & self) const {
    runtime_error("abs is not implemented for type %s", toString());
}
Tensor & Type::abs_(Tensor & self) const {
    runtime_error("abs_ is not implemented for type %s", toString());
}
Tensor & Type::sigmoid_(Tensor & self) const {
    runtime_error("sigmoid_ is not implemented for type %s", toString());
}
Tensor & Type::sigmoid_out(Tensor & result, const Tensor & self) const {
    runtime_error("sigmoid_out is not implemented for type %s", toString());
}
Tensor Type::sigmoid(const Tensor & self) const {
    runtime_error("sigmoid is not implemented for type %s", toString());
}
Tensor & Type::log_(Tensor & self) const {
    runtime_error("log_ is not implemented for type %s", toString());
}
Tensor & Type::log_out(Tensor & result, const Tensor & self) const {
    runtime_error("log_out is not implemented for type %s", toString());
}
Tensor Type::log(const Tensor & self) const {
    runtime_error("log is not implemented for type %s", toString());
}
Tensor & Type::log1p_(Tensor & self) const {
    runtime_error("log1p_ is not implemented for type %s", toString());
}
Tensor & Type::log1p_out(Tensor & result, const Tensor & self) const {
    runtime_error("log1p_out is not implemented for type %s", toString());
}
Tensor Type::log1p(const Tensor & self) const {
    runtime_error("log1p is not implemented for type %s", toString());
}
Tensor & Type::lgamma_out(Tensor & result, const Tensor & self) const {
    runtime_error("lgamma_out is not implemented for type %s", toString());
}
Tensor Type::lgamma(const Tensor & self) const {
    runtime_error("lgamma is not implemented for type %s", toString());
}
Tensor & Type::lgamma_(Tensor & self) const {
    runtime_error("lgamma_ is not implemented for type %s", toString());
}
Tensor & Type::exp_(Tensor & self) const {
    runtime_error("exp_ is not implemented for type %s", toString());
}
Tensor & Type::exp_out(Tensor & result, const Tensor & self) const {
    runtime_error("exp_out is not implemented for type %s", toString());
}
Tensor Type::exp(const Tensor & self) const {
    runtime_error("exp is not implemented for type %s", toString());
}
Tensor & Type::cos_(Tensor & self) const {
    runtime_error("cos_ is not implemented for type %s", toString());
}
Tensor & Type::cos_out(Tensor & result, const Tensor & self) const {
    runtime_error("cos_out is not implemented for type %s", toString());
}
Tensor Type::cos(const Tensor & self) const {
    runtime_error("cos is not implemented for type %s", toString());
}
Tensor & Type::acos_(Tensor & self) const {
    runtime_error("acos_ is not implemented for type %s", toString());
}
Tensor & Type::acos_out(Tensor & result, const Tensor & self) const {
    runtime_error("acos_out is not implemented for type %s", toString());
}
Tensor Type::acos(const Tensor & self) const {
    runtime_error("acos is not implemented for type %s", toString());
}
Tensor & Type::cosh_(Tensor & self) const {
    runtime_error("cosh_ is not implemented for type %s", toString());
}
Tensor & Type::cosh_out(Tensor & result, const Tensor & self) const {
    runtime_error("cosh_out is not implemented for type %s", toString());
}
Tensor Type::cosh(const Tensor & self) const {
    runtime_error("cosh is not implemented for type %s", toString());
}
Tensor & Type::sin_(Tensor & self) const {
    runtime_error("sin_ is not implemented for type %s", toString());
}
Tensor & Type::sin_out(Tensor & result, const Tensor & self) const {
    runtime_error("sin_out is not implemented for type %s", toString());
}
Tensor Type::sin(const Tensor & self) const {
    runtime_error("sin is not implemented for type %s", toString());
}
Tensor & Type::asin_(Tensor & self) const {
    runtime_error("asin_ is not implemented for type %s", toString());
}
Tensor & Type::asin_out(Tensor & result, const Tensor & self) const {
    runtime_error("asin_out is not implemented for type %s", toString());
}
Tensor Type::asin(const Tensor & self) const {
    runtime_error("asin is not implemented for type %s", toString());
}
Tensor & Type::sinh_(Tensor & self) const {
    runtime_error("sinh_ is not implemented for type %s", toString());
}
Tensor & Type::sinh_out(Tensor & result, const Tensor & self) const {
    runtime_error("sinh_out is not implemented for type %s", toString());
}
Tensor Type::sinh(const Tensor & self) const {
    runtime_error("sinh is not implemented for type %s", toString());
}
Tensor & Type::tan_(Tensor & self) const {
    runtime_error("tan_ is not implemented for type %s", toString());
}
Tensor & Type::tan_out(Tensor & result, const Tensor & self) const {
    runtime_error("tan_out is not implemented for type %s", toString());
}
Tensor Type::tan(const Tensor & self) const {
    runtime_error("tan is not implemented for type %s", toString());
}
Tensor & Type::atan_(Tensor & self) const {
    runtime_error("atan_ is not implemented for type %s", toString());
}
Tensor & Type::atan_out(Tensor & result, const Tensor & self) const {
    runtime_error("atan_out is not implemented for type %s", toString());
}
Tensor Type::atan(const Tensor & self) const {
    runtime_error("atan is not implemented for type %s", toString());
}
Tensor & Type::tanh_(Tensor & self) const {
    runtime_error("tanh_ is not implemented for type %s", toString());
}
Tensor & Type::tanh_out(Tensor & result, const Tensor & self) const {
    runtime_error("tanh_out is not implemented for type %s", toString());
}
Tensor Type::tanh(const Tensor & self) const {
    runtime_error("tanh is not implemented for type %s", toString());
}
Tensor & Type::erf_(Tensor & self) const {
    runtime_error("erf_ is not implemented for type %s", toString());
}
Tensor & Type::erf_out(Tensor & result, const Tensor & self) const {
    runtime_error("erf_out is not implemented for type %s", toString());
}
Tensor Type::erf(const Tensor & self) const {
    runtime_error("erf is not implemented for type %s", toString());
}
Tensor & Type::erfinv_(Tensor & self) const {
    runtime_error("erfinv_ is not implemented for type %s", toString());
}
Tensor & Type::erfinv_out(Tensor & result, const Tensor & self) const {
    runtime_error("erfinv_out is not implemented for type %s", toString());
}
Tensor Type::erfinv(const Tensor & self) const {
    runtime_error("erfinv is not implemented for type %s", toString());
}
Tensor & Type::sqrt_(Tensor & self) const {
    runtime_error("sqrt_ is not implemented for type %s", toString());
}
Tensor & Type::sqrt_out(Tensor & result, const Tensor & self) const {
    runtime_error("sqrt_out is not implemented for type %s", toString());
}
Tensor Type::sqrt(const Tensor & self) const {
    runtime_error("sqrt is not implemented for type %s", toString());
}
Tensor & Type::rsqrt_(Tensor & self) const {
    runtime_error("rsqrt_ is not implemented for type %s", toString());
}
Tensor & Type::rsqrt_out(Tensor & result, const Tensor & self) const {
    runtime_error("rsqrt_out is not implemented for type %s", toString());
}
Tensor Type::rsqrt(const Tensor & self) const {
    runtime_error("rsqrt is not implemented for type %s", toString());
}
Tensor & Type::ceil_(Tensor & self) const {
    runtime_error("ceil_ is not implemented for type %s", toString());
}
Tensor & Type::ceil_out(Tensor & result, const Tensor & self) const {
    runtime_error("ceil_out is not implemented for type %s", toString());
}
Tensor Type::ceil(const Tensor & self) const {
    runtime_error("ceil is not implemented for type %s", toString());
}
Tensor & Type::floor_(Tensor & self) const {
    runtime_error("floor_ is not implemented for type %s", toString());
}
Tensor & Type::floor_out(Tensor & result, const Tensor & self) const {
    runtime_error("floor_out is not implemented for type %s", toString());
}
Tensor Type::floor(const Tensor & self) const {
    runtime_error("floor is not implemented for type %s", toString());
}
Tensor & Type::round_(Tensor & self) const {
    runtime_error("round_ is not implemented for type %s", toString());
}
Tensor & Type::round_out(Tensor & result, const Tensor & self) const {
    runtime_error("round_out is not implemented for type %s", toString());
}
Tensor Type::round(const Tensor & self) const {
    runtime_error("round is not implemented for type %s", toString());
}
Tensor & Type::trunc_(Tensor & self) const {
    runtime_error("trunc_ is not implemented for type %s", toString());
}
Tensor & Type::trunc_out(Tensor & result, const Tensor & self) const {
    runtime_error("trunc_out is not implemented for type %s", toString());
}
Tensor Type::trunc(const Tensor & self) const {
    runtime_error("trunc is not implemented for type %s", toString());
}
Tensor & Type::frac_(Tensor & self) const {
    runtime_error("frac_ is not implemented for type %s", toString());
}
Tensor & Type::frac_out(Tensor & result, const Tensor & self) const {
    runtime_error("frac_out is not implemented for type %s", toString());
}
Tensor Type::frac(const Tensor & self) const {
    runtime_error("frac is not implemented for type %s", toString());
}
Tensor & Type::mean_out(Tensor & destination, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("mean_out is not implemented for type %s", toString());
}
Tensor Type::mean(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("mean is not implemented for type %s", toString());
}
Tensor Type::mean(const Tensor & self) const {
    runtime_error("mean is not implemented for type %s", toString());
}
Tensor & Type::var_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    runtime_error("var_out is not implemented for type %s", toString());
}
Tensor Type::var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    runtime_error("var is not implemented for type %s", toString());
}
Tensor Type::var(const Tensor & self, bool unbiased) const {
    runtime_error("var is not implemented for type %s", toString());
}
Tensor & Type::std_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    runtime_error("std_out is not implemented for type %s", toString());
}
Tensor Type::std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) const {
    runtime_error("std is not implemented for type %s", toString());
}
Tensor Type::std(const Tensor & self, bool unbiased) const {
    runtime_error("std is not implemented for type %s", toString());
}
Tensor & Type::norm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    runtime_error("norm_out is not implemented for type %s", toString());
}
Tensor Type::norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) const {
    runtime_error("norm is not implemented for type %s", toString());
}
Tensor Type::norm(const Tensor & self, Scalar p) const {
    runtime_error("norm is not implemented for type %s", toString());
}
Tensor & Type::renorm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    runtime_error("renorm_out is not implemented for type %s", toString());
}
Tensor Type::renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    runtime_error("renorm is not implemented for type %s", toString());
}
Tensor & Type::renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) const {
    runtime_error("renorm_ is not implemented for type %s", toString());
}
Tensor Type::s_dist(const Tensor & self, const Tensor & other, Scalar p) const {
    runtime_error("s_dist is not implemented for type %s", toString());
}
Tensor Type::dist(const Tensor & self, const Tensor & other, Scalar p) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "dist");
    return s_dist(b_self, b_other, p);
}
Tensor & Type::reciprocal_out(Tensor & destination, const Tensor & self) const {
    runtime_error("reciprocal_out is not implemented for type %s", toString());
}
Tensor Type::reciprocal(const Tensor & self) const {
    runtime_error("reciprocal is not implemented for type %s", toString());
}
Tensor & Type::reciprocal_(Tensor & self) const {
    runtime_error("reciprocal_ is not implemented for type %s", toString());
}
Tensor & Type::neg_out(Tensor & destination, const Tensor & self) const {
    runtime_error("neg_out is not implemented for type %s", toString());
}
Tensor Type::neg(const Tensor & self) const {
    runtime_error("neg is not implemented for type %s", toString());
}
Tensor & Type::neg_(Tensor & self) const {
    runtime_error("neg_ is not implemented for type %s", toString());
}
Tensor & Type::s_atan2_out(Tensor & destination, const Tensor & self, const Tensor & other) const {
    runtime_error("s_atan2_out is not implemented for type %s", toString());
}
Tensor & Type::atan2_out(Tensor & destination, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "atan2_out");
    return s_atan2_out(destination, b_self, b_other);
}
Tensor Type::s_atan2(const Tensor & self, const Tensor & other) const {
    runtime_error("s_atan2 is not implemented for type %s", toString());
}
Tensor Type::atan2(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "atan2");
    return s_atan2(b_self, b_other);
}
Tensor & Type::s_atan2_(Tensor & self, const Tensor & other) const {
    runtime_error("s_atan2_ is not implemented for type %s", toString());
}
Tensor & Type::atan2_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "atan2_");
    return s_atan2_(self, b_other);
}
Tensor & Type::pow_out(Tensor & destination, const Tensor & self, Scalar exponent) const {
    runtime_error("pow_out is not implemented for type %s", toString());
}
Tensor Type::pow(const Tensor & self, Scalar exponent) const {
    runtime_error("pow is not implemented for type %s", toString());
}
Tensor & Type::s_pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent) const {
    runtime_error("s_pow_out is not implemented for type %s", toString());
}
Tensor & Type::pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent) const {
    Tensor b_self, b_exponent;
    std::tie(b_self, b_exponent) = expand_outplace(self, exponent, "pow_out");
    return s_pow_out(destination, b_self, b_exponent);
}
Tensor Type::s_pow(const Tensor & self, const Tensor & exponent) const {
    runtime_error("s_pow is not implemented for type %s", toString());
}
Tensor Type::pow(const Tensor & self, const Tensor & exponent) const {
    Tensor b_self, b_exponent;
    std::tie(b_self, b_exponent) = expand_outplace(self, exponent, "pow");
    return s_pow(b_self, b_exponent);
}
Tensor & Type::pow_(Tensor & self, Scalar exponent) const {
    runtime_error("pow_ is not implemented for type %s", toString());
}
Tensor & Type::s_pow_(Tensor & self, const Tensor & exponent) const {
    runtime_error("s_pow_ is not implemented for type %s", toString());
}
Tensor & Type::pow_(Tensor & self, const Tensor & exponent) const {
    Tensor b_exponent;
    std::tie(b_exponent) = expand_inplace(self, exponent, "pow_");
    return s_pow_(self, b_exponent);
}
Tensor & Type::s_lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight) const {
    runtime_error("s_lerp_out is not implemented for type %s", toString());
}
Tensor & Type::lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_self, b_end;
    std::tie(b_self, b_end) = expand_outplace(self, end, "lerp_out");
    return s_lerp_out(destination, b_self, b_end, weight);
}
Tensor Type::s_lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    runtime_error("s_lerp is not implemented for type %s", toString());
}
Tensor Type::lerp(const Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_self, b_end;
    std::tie(b_self, b_end) = expand_outplace(self, end, "lerp");
    return s_lerp(b_self, b_end, weight);
}
Tensor & Type::s_lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    runtime_error("s_lerp_ is not implemented for type %s", toString());
}
Tensor & Type::lerp_(Tensor & self, const Tensor & end, Scalar weight) const {
    Tensor b_end;
    std::tie(b_end) = expand_inplace(self, end, "lerp_");
    return s_lerp_(self, b_end, weight);
}
Tensor & Type::linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    runtime_error("linspace_out is not implemented for type %s", toString());
}
Tensor Type::linspace(Scalar start, Scalar end, int64_t steps) const {
    runtime_error("linspace is not implemented for type %s", toString());
}
Tensor & Type::logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) const {
    runtime_error("logspace_out is not implemented for type %s", toString());
}
Tensor Type::logspace(Scalar start, Scalar end, int64_t steps) const {
    runtime_error("logspace is not implemented for type %s", toString());
}
Tensor & Type::histc_out(Tensor & destination, const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    runtime_error("histc_out is not implemented for type %s", toString());
}
Tensor Type::histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) const {
    runtime_error("histc is not implemented for type %s", toString());
}
Tensor & Type::zero_(Tensor & self) const {
    runtime_error("zero_ is not implemented for type %s", toString());
}
Tensor & Type::sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("sum_out is not implemented for type %s", toString());
}
Tensor Type::sum(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("sum is not implemented for type %s", toString());
}
Tensor Type::sum(const Tensor & self) const {
    runtime_error("sum is not implemented for type %s", toString());
}
Tensor & Type::prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("prod_out is not implemented for type %s", toString());
}
Tensor Type::prod(const Tensor & self, int64_t dim, bool keepdim) const {
    runtime_error("prod is not implemented for type %s", toString());
}
Tensor Type::prod(const Tensor & self) const {
    runtime_error("prod is not implemented for type %s", toString());
}
Tensor & Type::cumsum_out(Tensor & result, const Tensor & self, int64_t dim) const {
    runtime_error("cumsum_out is not implemented for type %s", toString());
}
Tensor Type::cumsum(const Tensor & self, int64_t dim) const {
    runtime_error("cumsum is not implemented for type %s", toString());
}
Tensor & Type::cumprod_out(Tensor & result, const Tensor & self, int64_t dim) const {
    runtime_error("cumprod_out is not implemented for type %s", toString());
}
Tensor Type::cumprod(const Tensor & self, int64_t dim) const {
    runtime_error("cumprod is not implemented for type %s", toString());
}
Tensor & Type::sign_out(Tensor & result, const Tensor & self) const {
    runtime_error("sign_out is not implemented for type %s", toString());
}
Tensor Type::sign(const Tensor & self) const {
    runtime_error("sign is not implemented for type %s", toString());
}
Tensor & Type::sign_(Tensor & self) const {
    runtime_error("sign_ is not implemented for type %s", toString());
}
Tensor Type::trace(const Tensor & self) const {
    runtime_error("trace is not implemented for type %s", toString());
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("add_out is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("add is not implemented for type %s", toString());
}
Tensor & Type::s_add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_add_out is not implemented for type %s", toString());
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add_out");
    return s_add_out(result, b_self, b_other, alpha);
}
Tensor Type::s_add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_add is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "add");
    return s_add(b_self, b_other, alpha);
}
Tensor & Type::add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) const {
    runtime_error("add_out is not implemented for type %s", toString());
}
Tensor Type::add(const Tensor & self, SparseTensor other, Scalar alpha) const {
    runtime_error("add is not implemented for type %s", toString());
}
Tensor & Type::add_(Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("add_ is not implemented for type %s", toString());
}
Tensor & Type::s_add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_add_ is not implemented for type %s", toString());
}
Tensor & Type::add_(Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "add_");
    return s_add_(self, b_other, alpha);
}
Tensor & Type::add_(Tensor & self, SparseTensor other, Scalar alpha) const {
    runtime_error("add_ is not implemented for type %s", toString());
}
Tensor & Type::sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("sub_out is not implemented for type %s", toString());
}
Tensor Type::sub(const Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("sub is not implemented for type %s", toString());
}
Tensor & Type::s_sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_sub_out is not implemented for type %s", toString());
}
Tensor & Type::sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub_out");
    return s_sub_out(result, b_self, b_other, alpha);
}
Tensor Type::s_sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_sub is not implemented for type %s", toString());
}
Tensor Type::sub(const Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "sub");
    return s_sub(b_self, b_other, alpha);
}
Tensor & Type::sub_(Tensor & self, Scalar other, Scalar alpha) const {
    runtime_error("sub_ is not implemented for type %s", toString());
}
Tensor & Type::s_sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    runtime_error("s_sub_ is not implemented for type %s", toString());
}
Tensor & Type::sub_(Tensor & self, const Tensor & other, Scalar alpha) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "sub_");
    return s_sub_(self, b_other, alpha);
}
Tensor & Type::mul_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("mul_out is not implemented for type %s", toString());
}
Tensor Type::mul(const Tensor & self, Scalar other) const {
    runtime_error("mul is not implemented for type %s", toString());
}
Tensor & Type::s_mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_mul_out is not implemented for type %s", toString());
}
Tensor & Type::mul_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul_out");
    return s_mul_out(result, b_self, b_other);
}
Tensor Type::s_mul(const Tensor & self, const Tensor & other) const {
    runtime_error("s_mul is not implemented for type %s", toString());
}
Tensor Type::mul(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "mul");
    return s_mul(b_self, b_other);
}
Tensor & Type::mul_(Tensor & self, Scalar other) const {
    runtime_error("mul_ is not implemented for type %s", toString());
}
Tensor & Type::s_mul_(Tensor & self, const Tensor & other) const {
    runtime_error("s_mul_ is not implemented for type %s", toString());
}
Tensor & Type::mul_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "mul_");
    return s_mul_(self, b_other);
}
Tensor & Type::div_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("div_out is not implemented for type %s", toString());
}
Tensor Type::div(const Tensor & self, Scalar other) const {
    runtime_error("div is not implemented for type %s", toString());
}
Tensor & Type::s_div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_div_out is not implemented for type %s", toString());
}
Tensor & Type::div_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "div_out");
    return s_div_out(result, b_self, b_other);
}
Tensor Type::s_div(const Tensor & self, const Tensor & other) const {
    runtime_error("s_div is not implemented for type %s", toString());
}
Tensor Type::div(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "div");
    return s_div(b_self, b_other);
}
Tensor & Type::div_(Tensor & self, Scalar other) const {
    runtime_error("div_ is not implemented for type %s", toString());
}
Tensor & Type::s_div_(Tensor & self, const Tensor & other) const {
    runtime_error("s_div_ is not implemented for type %s", toString());
}
Tensor & Type::div_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "div_");
    return s_div_(self, b_other);
}
Tensor & Type::fmod_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("fmod_out is not implemented for type %s", toString());
}
Tensor Type::fmod(const Tensor & self, Scalar other) const {
    runtime_error("fmod is not implemented for type %s", toString());
}
Tensor & Type::s_fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_fmod_out is not implemented for type %s", toString());
}
Tensor & Type::fmod_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "fmod_out");
    return s_fmod_out(result, b_self, b_other);
}
Tensor Type::s_fmod(const Tensor & self, const Tensor & other) const {
    runtime_error("s_fmod is not implemented for type %s", toString());
}
Tensor Type::fmod(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "fmod");
    return s_fmod(b_self, b_other);
}
Tensor & Type::fmod_(Tensor & self, Scalar other) const {
    runtime_error("fmod_ is not implemented for type %s", toString());
}
Tensor & Type::s_fmod_(Tensor & self, const Tensor & other) const {
    runtime_error("s_fmod_ is not implemented for type %s", toString());
}
Tensor & Type::fmod_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "fmod_");
    return s_fmod_(self, b_other);
}
Tensor & Type::remainder_out(Tensor & result, const Tensor & self, Scalar other) const {
    runtime_error("remainder_out is not implemented for type %s", toString());
}
Tensor Type::remainder(const Tensor & self, Scalar other) const {
    runtime_error("remainder is not implemented for type %s", toString());
}
Tensor & Type::s_remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    runtime_error("s_remainder_out is not implemented for type %s", toString());
}
Tensor & Type::remainder_out(Tensor & result, const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "remainder_out");
    return s_remainder_out(result, b_self, b_other);
}
Tensor Type::s_remainder(const Tensor & self, const Tensor & other) const {
    runtime_error("s_remainder is not implemented for type %s", toString());
}
Tensor Type::remainder(const Tensor & self, const Tensor & other) const {
    Tensor b_self, b_other;
    std::tie(b_self, b_other) = expand_outplace(self, other, "remainder");
    return s_remainder(b_self, b_other);
}
Tensor & Type::remainder_(Tensor & self, Scalar other) const {
    runtime_error("remainder_ is not implemented for type %s", toString());
}
Tensor & Type::s_remainder_(Tensor & self, const Tensor & other) const {
    runtime_error("s_remainder_ is not implemented for type %s", toString());
}
Tensor & Type::remainder_(Tensor & self, const Tensor & other) const {
    Tensor b_other;
    std::tie(b_other) = expand_inplace(self, other, "remainder_");
    return s_remainder_(self, b_other);
}
Tensor & Type::clamp_out(Tensor & destination, const Tensor & self, Scalar min, Scalar max) const {
    runtime_error("clamp_out is not implemented for type %s", toString());
}
Tensor Type::clamp(const Tensor & self, Scalar min, Scalar max) const {
    runtime_error("clamp is not implemented for type %s", toString());
}
Tensor & Type::clamp_(Tensor & self, Scalar min, Scalar max) const {
    runtime_error("clamp_ is not implemented for type %s", toString());
}
Tensor & Type::clamp_min_out(Tensor & result, const Tensor & self, Scalar min) const {
    runtime_error("clamp_min_out is not implemented for type %s", toString());
}
Tensor Type::clamp_min(const Tensor & self, Scalar min) const {
    runtime_error("clamp_min is not implemented for type %s", toString());
}
Tensor & Type::clamp_min_(Tensor & self, Scalar min) const {
    runtime_error("clamp_min_ is not implemented for type %s", toString());
}
Tensor & Type::clamp_max_out(Tensor & result, const Tensor & self, Scalar max) const {
    runtime_error("clamp_max_out is not implemented for type %s", toString());
}
Tensor Type::clamp_max(const Tensor & self, Scalar max) const {
    runtime_error("clamp_max is not implemented for type %s", toString());
}
Tensor & Type::clamp_max_(Tensor & self, Scalar max) const {
    runtime_error("clamp_max_ is not implemented for type %s", toString());
}
Tensor Type::dot(const Tensor & self, const Tensor & tensor) const {
    runtime_error("dot is not implemented for type %s", toString());
}
Tensor & Type::tril_out(Tensor & destination, const Tensor & self, int64_t diagonal) const {
    runtime_error("tril_out is not implemented for type %s", toString());
}
Tensor Type::tril(const Tensor & self, int64_t diagonal) const {
    runtime_error("tril is not implemented for type %s", toString());
}
Tensor & Type::tril_(Tensor & self, int64_t diagonal) const {
    runtime_error("tril_ is not implemented for type %s", toString());
}
Tensor & Type::triu_out(Tensor & destination, const Tensor & self, int64_t diagonal) const {
    runtime_error("triu_out is not implemented for type %s", toString());
}
Tensor Type::triu(const Tensor & self, int64_t diagonal) const {
    runtime_error("triu is not implemented for type %s", toString());
}
Tensor & Type::triu_(Tensor & self, int64_t diagonal) const {
    runtime_error("triu_ is not implemented for type %s", toString());
}
Tensor & Type::cross_out(Tensor & destination, const Tensor & self, const Tensor & other, int64_t dim) const {
    runtime_error("cross_out is not implemented for type %s", toString());
}
Tensor Type::cross(const Tensor & self, const Tensor & other, int64_t dim) const {
    runtime_error("cross is not implemented for type %s", toString());
}
Tensor & Type::eye_out(Tensor & result, int64_t n, int64_t m) const {
    runtime_error("eye_out is not implemented for type %s", toString());
}
Tensor Type::eye(int64_t n, int64_t m) const {
    runtime_error("eye is not implemented for type %s", toString());
}
Tensor & Type::diag_out(Tensor & result, const Tensor & self, int64_t diagonal) const {
    runtime_error("diag_out is not implemented for type %s", toString());
}
Tensor Type::diag(const Tensor & self, int64_t diagonal) const {
    runtime_error("diag is not implemented for type %s", toString());
}
Tensor & Type::s_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addmm_out is not implemented for type %s", toString());
}
Tensor & Type::addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0),mat2.size(1)}, "addmm_out");
    return s_addmm_out(result, b_self, mat1, mat2, beta, alpha);
}
Tensor Type::s_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addmm is not implemented for type %s", toString());
}
Tensor Type::addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat1.size(0),mat2.size(1)}, "addmm");
    return s_addmm(b_self, mat1, mat2, beta, alpha);
}
Tensor & Type::addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) const {
    runtime_error("addmm_ is not implemented for type %s", toString());
}
Tensor & Type::s_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    runtime_error("s_addmv_out is not implemented for type %s", toString());
}
Tensor & Type::addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat.size(0)}, "addmv_out");
    return s_addmv_out(result, b_self, mat, vec, beta, alpha);
}
Tensor Type::s_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    runtime_error("s_addmv is not implemented for type %s", toString());
}
Tensor Type::addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {mat.size(0)}, "addmv");
    return s_addmv(b_self, mat, vec, beta, alpha);
}
Tensor & Type::addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) const {
    runtime_error("addmv_ is not implemented for type %s", toString());
}
Tensor & Type::s_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addr_out is not implemented for type %s", toString());
}
Tensor & Type::addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {vec1.size(0),vec2.size(0)}, "addr_out");
    return s_addr_out(result, b_self, vec1, vec2, beta, alpha);
}
Tensor Type::s_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addr is not implemented for type %s", toString());
}
Tensor Type::addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {vec1.size(0),vec2.size(0)}, "addr");
    return s_addr(b_self, vec1, vec2, beta, alpha);
}
Tensor & Type::addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) const {
    runtime_error("addr_ is not implemented for type %s", toString());
}
Tensor & Type::ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) const {
    runtime_error("ger_out is not implemented for type %s", toString());
}
Tensor Type::ger(const Tensor & self, const Tensor & vec2) const {
    runtime_error("ger is not implemented for type %s", toString());
}
Tensor & Type::mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    runtime_error("mv_out is not implemented for type %s", toString());
}
Tensor Type::mv(const Tensor & self, const Tensor & vec) const {
    runtime_error("mv is not implemented for type %s", toString());
}
Tensor & Type::mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    runtime_error("mm_out is not implemented for type %s", toString());
}
Tensor Type::mm(const Tensor & self, const Tensor & mat2) const {
    runtime_error("mm is not implemented for type %s", toString());
}
Tensor & Type::bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    runtime_error("bmm_out is not implemented for type %s", toString());
}
Tensor Type::bmm(const Tensor & self, const Tensor & mat2) const {
    runtime_error("bmm is not implemented for type %s", toString());
}
Tensor & Type::s_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addbmm_out is not implemented for type %s", toString());
}
Tensor & Type::addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(1),batch2.size(2)}, "addbmm_out");
    return s_addbmm_out(result, b_self, batch1, batch2, beta, alpha);
}
Tensor Type::s_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("s_addbmm is not implemented for type %s", toString());
}
Tensor Type::addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(1),batch2.size(2)}, "addbmm");
    return s_addbmm(b_self, batch1, batch2, beta, alpha);
}
Tensor & Type::addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("addbmm_ is not implemented for type %s", toString());
}
Tensor & Type::s_baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("s_baddbmm_out is not implemented for type %s", toString());
}
Tensor & Type::baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(0),batch1.size(1),batch2.size(2)}, "baddbmm_out");
    return s_baddbmm_out(result, b_self, batch1, batch2, beta, alpha);
}
Tensor Type::s_baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("s_baddbmm is not implemented for type %s", toString());
}
Tensor Type::baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    Tensor b_self;
    std::tie(b_self) = expand_size(self, {batch1.size(0),batch1.size(1),batch2.size(2)}, "baddbmm");
    return s_baddbmm(b_self, batch1, batch2, beta, alpha);
}
Tensor & Type::baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) const {
    runtime_error("baddbmm_ is not implemented for type %s", toString());
}
Tensor & Type::s_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcmul_out is not implemented for type %s", toString());
}
Tensor & Type::addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcmul_out");
    return s_addcmul_out(result, b_self, b_tensor1, b_tensor2, value);
}
Tensor Type::s_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcmul is not implemented for type %s", toString());
}
Tensor Type::addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcmul");
    return s_addcmul(b_self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcmul_ is not implemented for type %s", toString());
}
Tensor & Type::addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_tensor1, b_tensor2;
    std::tie(b_tensor1, b_tensor2) = expand_inplace(self, tensor1, tensor2, "addcmul_");
    return s_addcmul_(self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcdiv_out is not implemented for type %s", toString());
}
Tensor & Type::addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcdiv_out");
    return s_addcdiv_out(result, b_self, b_tensor1, b_tensor2, value);
}
Tensor Type::s_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcdiv is not implemented for type %s", toString());
}
Tensor Type::addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_self, b_tensor1, b_tensor2;
    std::tie(b_self, b_tensor1, b_tensor2) = expand_outplace(self, tensor1, tensor2, "addcdiv");
    return s_addcdiv(b_self, b_tensor1, b_tensor2, value);
}
Tensor & Type::s_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    runtime_error("s_addcdiv_ is not implemented for type %s", toString());
}
Tensor & Type::addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) const {
    Tensor b_tensor1, b_tensor2;
    std::tie(b_tensor1, b_tensor2) = expand_inplace(self, tensor1, tensor2, "addcdiv_");
    return s_addcdiv_(self, b_tensor1, b_tensor2, value);
}
std::tuple<Tensor &,Tensor &> Type::gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) const {
    runtime_error("gesv_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::gesv(const Tensor & self, const Tensor & A) const {
    runtime_error("gesv is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) const {
    runtime_error("gels_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::gels(const Tensor & self, const Tensor & A) const {
    runtime_error("gels is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    runtime_error("trtrs_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) const {
    runtime_error("trtrs is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) const {
    runtime_error("symeig_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::symeig(const Tensor & self, bool eigenvectors, bool upper) const {
    runtime_error("symeig is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) const {
    runtime_error("eig_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::eig(const Tensor & self, bool eigenvectors) const {
    runtime_error("eig is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) const {
    runtime_error("svd_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::svd(const Tensor & self, bool some) const {
    runtime_error("svd is not implemented for type %s", toString());
}
Tensor & Type::inverse_out(Tensor & output, const Tensor & self) const {
    runtime_error("inverse_out is not implemented for type %s", toString());
}
Tensor Type::inverse(const Tensor & self) const {
    runtime_error("inverse is not implemented for type %s", toString());
}
Tensor & Type::potrf_out(Tensor & output, const Tensor & self, bool upper) const {
    runtime_error("potrf_out is not implemented for type %s", toString());
}
Tensor Type::potrf(const Tensor & self, bool upper) const {
    runtime_error("potrf is not implemented for type %s", toString());
}
Tensor & Type::potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) const {
    runtime_error("potrs_out is not implemented for type %s", toString());
}
Tensor Type::potrs(const Tensor & self, const Tensor & input2, bool upper) const {
    runtime_error("potrs is not implemented for type %s", toString());
}
Tensor & Type::potri_out(Tensor & output, const Tensor & self, bool upper) const {
    runtime_error("potri_out is not implemented for type %s", toString());
}
Tensor Type::potri(const Tensor & self, bool upper) const {
    runtime_error("potri is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) const {
    runtime_error("pstrf_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::pstrf(const Tensor & self, bool upper, Scalar tol) const {
    runtime_error("pstrf is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::qr_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    runtime_error("qr_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::qr(const Tensor & self) const {
    runtime_error("qr is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) const {
    runtime_error("geqrf_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::geqrf(const Tensor & self) const {
    runtime_error("geqrf is not implemented for type %s", toString());
}
Tensor & Type::orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) const {
    runtime_error("orgqr_out is not implemented for type %s", toString());
}
Tensor Type::orgqr(const Tensor & self, const Tensor & input2) const {
    runtime_error("orgqr is not implemented for type %s", toString());
}
Tensor & Type::ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    runtime_error("ormqr_out is not implemented for type %s", toString());
}
Tensor Type::ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) const {
    runtime_error("ormqr is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, const Tensor & info, bool pivot) const {
    runtime_error("btrifact_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::btrifact(const Tensor & self, const Tensor & info, bool pivot) const {
    runtime_error("btrifact is not implemented for type %s", toString());
}
Tensor & Type::btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    runtime_error("btrisolve_out is not implemented for type %s", toString());
}
Tensor Type::btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) const {
    runtime_error("btrisolve is not implemented for type %s", toString());
}
Tensor & Type::randperm_out(Tensor & result, int64_t n, Generator * generator) const {
    runtime_error("randperm_out is not implemented for type %s", toString());
}
Tensor Type::randperm(int64_t n, Generator * generator) const {
    runtime_error("randperm is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, int64_t from, int64_t to, Generator * generator) const {
    runtime_error("random_ is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, int64_t to, Generator * generator) const {
    runtime_error("random_ is not implemented for type %s", toString());
}
Tensor & Type::random_(Tensor & self, Generator * generator) const {
    runtime_error("random_ is not implemented for type %s", toString());
}
Tensor & Type::multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    runtime_error("multinomial_out is not implemented for type %s", toString());
}
Tensor Type::multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) const {
    runtime_error("multinomial is not implemented for type %s", toString());
}
Tensor & Type::uniform_(Tensor & self, double from, double to, Generator * generator) const {
    runtime_error("uniform_ is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, const Tensor & means, double std, Generator * generator) const {
    runtime_error("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(const Tensor & means, double std, Generator * generator) const {
    runtime_error("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) const {
    runtime_error("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(double mean, const Tensor & std, Generator * generator) const {
    runtime_error("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_out(Tensor & output, const Tensor & means, const Tensor & std, Generator * generator) const {
    runtime_error("normal_out is not implemented for type %s", toString());
}
Tensor Type::normal(const Tensor & means, const Tensor & std, Generator * generator) const {
    runtime_error("normal is not implemented for type %s", toString());
}
Tensor & Type::normal_(Tensor & self, double mean, double std, Generator * generator) const {
    runtime_error("normal_ is not implemented for type %s", toString());
}
Tensor & Type::cauchy_(Tensor & self, double median, double sigma, Generator * generator) const {
    runtime_error("cauchy_ is not implemented for type %s", toString());
}
Tensor & Type::log_normal_(Tensor & self, double mean, double std, Generator * generator) const {
    runtime_error("log_normal_ is not implemented for type %s", toString());
}
Tensor & Type::exponential_(Tensor & self, double lambd, Generator * generator) const {
    runtime_error("exponential_ is not implemented for type %s", toString());
}
Tensor & Type::rand_out(Tensor & result, IntList size, Generator * generator) const {
    runtime_error("rand_out is not implemented for type %s", toString());
}
Tensor Type::rand(IntList size, Generator * generator) const {
    runtime_error("rand is not implemented for type %s", toString());
}
Tensor & Type::randn_out(Tensor & result, IntList size, Generator * generator) const {
    runtime_error("randn_out is not implemented for type %s", toString());
}
Tensor Type::randn(IntList size, Generator * generator) const {
    runtime_error("randn is not implemented for type %s", toString());
}
Tensor & Type::geometric_(Tensor & self, double p, Generator * generator) const {
    runtime_error("geometric_ is not implemented for type %s", toString());
}
Tensor Type::tensor(Storage & storage, int64_t storageOffset, IntList size, IntList stride) const {
    runtime_error("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor(IntList size) const {
    runtime_error("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor(IntList size, IntList stride) const {
    runtime_error("tensor is not implemented for type %s", toString());
}
Tensor Type::tensor() const {
    runtime_error("tensor is not implemented for type %s", toString());
}
Tensor Type::alias(const Tensor & self) const {
    runtime_error("alias is not implemented for type %s", toString());
}
Tensor & Type::assign_(Tensor & self, const Tensor & src) const {
    runtime_error("assign_ is not implemented for type %s", toString());
}
Tensor & Type::as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    runtime_error("as_strided_out is not implemented for type %s", toString());
}
Tensor Type::as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    runtime_error("as_strided is not implemented for type %s", toString());
}
Tensor & Type::as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) const {
    runtime_error("as_strided_ is not implemented for type %s", toString());
}
Tensor & Type::cat_out(Tensor & self, TensorList tensors, int64_t dim) const {
    runtime_error("cat_out is not implemented for type %s", toString());
}
Tensor Type::cat(TensorList tensors, int64_t dim) const {
    runtime_error("cat is not implemented for type %s", toString());
}
Tensor & Type::reshape_(Tensor & self, IntList size, IntList stride) const {
    runtime_error("reshape_ is not implemented for type %s", toString());
}
Tensor & Type::binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy_out is not implemented for type %s", toString());
}
Tensor Type::binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy is not implemented for type %s", toString());
}
Tensor & Type::binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy_forward_out is not implemented for type %s", toString());
}
Tensor Type::binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy_forward is not implemented for type %s", toString());
}
Tensor & Type::binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy_backward_out is not implemented for type %s", toString());
}
Tensor Type::binary_cross_entropy_backward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average) const {
    runtime_error("binary_cross_entropy_backward is not implemented for type %s", toString());
}
Tensor & Type::kl_div_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div_out is not implemented for type %s", toString());
}
Tensor Type::kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div is not implemented for type %s", toString());
}
Tensor & Type::kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div_forward_out is not implemented for type %s", toString());
}
Tensor Type::kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div_forward is not implemented for type %s", toString());
}
Tensor & Type::kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div_backward_out is not implemented for type %s", toString());
}
Tensor Type::kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("kl_div_backward is not implemented for type %s", toString());
}
Tensor & Type::l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss_out is not implemented for type %s", toString());
}
Tensor Type::l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss is not implemented for type %s", toString());
}
Tensor & Type::l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("l1_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss_out is not implemented for type %s", toString());
}
Tensor Type::mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss is not implemented for type %s", toString());
}
Tensor & Type::mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("mse_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss_out is not implemented for type %s", toString());
}
Tensor Type::multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss is not implemented for type %s", toString());
}
Tensor & Type::multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::multi_margin_loss_backward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) const {
    runtime_error("multi_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("multilabel_margin_loss_out is not implemented for type %s", toString());
}
Tensor Type::multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("multilabel_margin_loss is not implemented for type %s", toString());
}
Tensor & Type::multilabel_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, const Tensor & is_target) const {
    runtime_error("multilabel_margin_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, const Tensor & is_target) const {
    runtime_error("multilabel_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, bool size_average, const Tensor & is_target) const {
    runtime_error("multilabel_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::multilabel_margin_loss_backward(const Tensor & self, const Tensor & target, bool size_average, const Tensor & is_target) const {
    runtime_error("multilabel_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    runtime_error("nll_loss_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    runtime_error("nll_loss is not implemented for type %s", toString());
}
Tensor & Type::nll_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    runtime_error("nll_loss2d_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) const {
    runtime_error("nll_loss2d is not implemented for type %s", toString());
}
Tensor & Type::nll_loss2d_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss2d_forward is not implemented for type %s", toString());
}
Tensor & Type::nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) const {
    runtime_error("nll_loss2d_backward is not implemented for type %s", toString());
}
Tensor & Type::smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss_out is not implemented for type %s", toString());
}
Tensor Type::smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss is not implemented for type %s", toString());
}
Tensor & Type::smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) const {
    runtime_error("smooth_l1_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss_out is not implemented for type %s", toString());
}
Tensor Type::soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss is not implemented for type %s", toString());
}
Tensor & Type::soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss_forward_out is not implemented for type %s", toString());
}
Tensor Type::soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss_forward is not implemented for type %s", toString());
}
Tensor & Type::soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss_backward_out is not implemented for type %s", toString());
}
Tensor Type::soft_margin_loss_backward(const Tensor & self, const Tensor & target, bool size_average) const {
    runtime_error("soft_margin_loss_backward is not implemented for type %s", toString());
}
Tensor & Type::elu_out(Tensor & output, const Tensor & self, Scalar alpha) const {
    runtime_error("elu_out is not implemented for type %s", toString());
}
Tensor Type::elu(const Tensor & self, Scalar alpha) const {
    runtime_error("elu is not implemented for type %s", toString());
}
Tensor & Type::elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha) const {
    runtime_error("elu_forward_out is not implemented for type %s", toString());
}
Tensor Type::elu_forward(const Tensor & self, Scalar alpha) const {
    runtime_error("elu_forward is not implemented for type %s", toString());
}
Tensor & Type::elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, const Tensor & output) const {
    runtime_error("elu_backward_out is not implemented for type %s", toString());
}
Tensor Type::elu_backward(const Tensor & grad_output, Scalar alpha, const Tensor & output) const {
    runtime_error("elu_backward is not implemented for type %s", toString());
}
Tensor Type::elu_(Tensor & self, Scalar alpha) const {
    runtime_error("elu_ is not implemented for type %s", toString());
}
Tensor Type::elu_forward_(Tensor & self, Scalar alpha) const {
    runtime_error("elu_forward_ is not implemented for type %s", toString());
}
Tensor & Type::glu_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("glu_out is not implemented for type %s", toString());
}
Tensor Type::glu(const Tensor & self, int64_t dim) const {
    runtime_error("glu is not implemented for type %s", toString());
}
Tensor & Type::glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("glu_forward_out is not implemented for type %s", toString());
}
Tensor Type::glu_forward(const Tensor & self, int64_t dim) const {
    runtime_error("glu_forward is not implemented for type %s", toString());
}
Tensor & Type::glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    runtime_error("glu_backward_out is not implemented for type %s", toString());
}
Tensor Type::glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) const {
    runtime_error("glu_backward is not implemented for type %s", toString());
}
Tensor & Type::hardshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink_out is not implemented for type %s", toString());
}
Tensor Type::hardshrink(const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink is not implemented for type %s", toString());
}
Tensor & Type::hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink_forward_out is not implemented for type %s", toString());
}
Tensor Type::hardshrink_forward(const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink_forward is not implemented for type %s", toString());
}
Tensor & Type::hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink_backward_out is not implemented for type %s", toString());
}
Tensor Type::hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    runtime_error("hardshrink_backward is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_out is not implemented for type %s", toString());
}
Tensor Type::hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_forward_out is not implemented for type %s", toString());
}
Tensor Type::hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_forward is not implemented for type %s", toString());
}
Tensor & Type::hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_backward_out is not implemented for type %s", toString());
}
Tensor Type::hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_backward is not implemented for type %s", toString());
}
Tensor Type::hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_ is not implemented for type %s", toString());
}
Tensor Type::hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) const {
    runtime_error("hardtanh_forward_ is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_out is not implemented for type %s", toString());
}
Tensor Type::leaky_relu(const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_forward_out is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_forward(const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_forward is not implemented for type %s", toString());
}
Tensor & Type::leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_backward_out is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_backward is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_(Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_ is not implemented for type %s", toString());
}
Tensor Type::leaky_relu_forward_(Tensor & self, Scalar negative_slope) const {
    runtime_error("leaky_relu_forward_ is not implemented for type %s", toString());
}
Tensor & Type::log_sigmoid_out(Tensor & output, const Tensor & self) const {
    runtime_error("log_sigmoid_out is not implemented for type %s", toString());
}
Tensor Type::log_sigmoid(const Tensor & self) const {
    runtime_error("log_sigmoid is not implemented for type %s", toString());
}
Tensor & Type::log_sigmoid_forward_out(Tensor & output, const Tensor & self, const Tensor & buffer) const {
    runtime_error("log_sigmoid_forward_out is not implemented for type %s", toString());
}
Tensor Type::log_sigmoid_forward(const Tensor & self, const Tensor & buffer) const {
    runtime_error("log_sigmoid_forward is not implemented for type %s", toString());
}
Tensor & Type::log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    runtime_error("log_sigmoid_backward_out is not implemented for type %s", toString());
}
Tensor Type::log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) const {
    runtime_error("log_sigmoid_backward is not implemented for type %s", toString());
}
Tensor & Type::log_softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("log_softmax_out is not implemented for type %s", toString());
}
Tensor Type::log_softmax(const Tensor & self, int64_t dim) const {
    runtime_error("log_softmax is not implemented for type %s", toString());
}
Tensor & Type::log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("log_softmax_forward_out is not implemented for type %s", toString());
}
Tensor Type::log_softmax_forward(const Tensor & self, int64_t dim) const {
    runtime_error("log_softmax_forward is not implemented for type %s", toString());
}
Tensor & Type::log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    runtime_error("log_softmax_backward_out is not implemented for type %s", toString());
}
Tensor Type::log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    runtime_error("log_softmax_backward is not implemented for type %s", toString());
}
Tensor & Type::prelu_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    runtime_error("prelu_out is not implemented for type %s", toString());
}
Tensor Type::prelu(const Tensor & self, const Tensor & weight) const {
    runtime_error("prelu is not implemented for type %s", toString());
}
Tensor & Type::prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) const {
    runtime_error("prelu_forward_out is not implemented for type %s", toString());
}
Tensor Type::prelu_forward(const Tensor & self, const Tensor & weight) const {
    runtime_error("prelu_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) const {
    runtime_error("prelu_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) const {
    runtime_error("prelu_backward is not implemented for type %s", toString());
}
Tensor & Type::rrelu_out(Tensor & output, const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    runtime_error("rrelu_out is not implemented for type %s", toString());
}
Tensor Type::rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    runtime_error("rrelu is not implemented for type %s", toString());
}
Tensor & Type::rrelu_forward_out(Tensor & output, const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator, const Tensor & noise) const {
    runtime_error("rrelu_forward_out is not implemented for type %s", toString());
}
Tensor Type::rrelu_forward(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator, const Tensor & noise) const {
    runtime_error("rrelu_forward is not implemented for type %s", toString());
}
Tensor & Type::rrelu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lower, Scalar upper, bool training, const Tensor & noise) const {
    runtime_error("rrelu_backward_out is not implemented for type %s", toString());
}
Tensor Type::rrelu_backward(const Tensor & grad_output, const Tensor & self, Scalar lower, Scalar upper, bool training, const Tensor & noise) const {
    runtime_error("rrelu_backward is not implemented for type %s", toString());
}
Tensor Type::rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) const {
    runtime_error("rrelu_ is not implemented for type %s", toString());
}
Tensor Type::rrelu_forward_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator, const Tensor & noise) const {
    runtime_error("rrelu_forward_ is not implemented for type %s", toString());
}
Tensor & Type::softmax_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("softmax_out is not implemented for type %s", toString());
}
Tensor Type::softmax(const Tensor & self, int64_t dim) const {
    runtime_error("softmax is not implemented for type %s", toString());
}
Tensor & Type::softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) const {
    runtime_error("softmax_forward_out is not implemented for type %s", toString());
}
Tensor Type::softmax_forward(const Tensor & self, int64_t dim) const {
    runtime_error("softmax_forward is not implemented for type %s", toString());
}
Tensor & Type::softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    runtime_error("softmax_backward_out is not implemented for type %s", toString());
}
Tensor Type::softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) const {
    runtime_error("softmax_backward is not implemented for type %s", toString());
}
Tensor & Type::softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    runtime_error("softplus_out is not implemented for type %s", toString());
}
Tensor Type::softplus(const Tensor & self, Scalar beta, Scalar threshold) const {
    runtime_error("softplus is not implemented for type %s", toString());
}
Tensor & Type::softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) const {
    runtime_error("softplus_forward_out is not implemented for type %s", toString());
}
Tensor Type::softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) const {
    runtime_error("softplus_forward is not implemented for type %s", toString());
}
Tensor & Type::softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    runtime_error("softplus_backward_out is not implemented for type %s", toString());
}
Tensor Type::softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) const {
    runtime_error("softplus_backward is not implemented for type %s", toString());
}
Tensor & Type::softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink_out is not implemented for type %s", toString());
}
Tensor Type::softshrink(const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink is not implemented for type %s", toString());
}
Tensor & Type::softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink_forward_out is not implemented for type %s", toString());
}
Tensor Type::softshrink_forward(const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink_forward is not implemented for type %s", toString());
}
Tensor & Type::softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink_backward_out is not implemented for type %s", toString());
}
Tensor Type::softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) const {
    runtime_error("softshrink_backward is not implemented for type %s", toString());
}
Tensor & Type::threshold_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_out is not implemented for type %s", toString());
}
Tensor Type::threshold(const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold is not implemented for type %s", toString());
}
Tensor & Type::threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_forward_out is not implemented for type %s", toString());
}
Tensor Type::threshold_forward(const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_forward is not implemented for type %s", toString());
}
Tensor & Type::threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_backward_out is not implemented for type %s", toString());
}
Tensor Type::threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_backward is not implemented for type %s", toString());
}
Tensor Type::threshold_(Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_ is not implemented for type %s", toString());
}
Tensor Type::threshold_forward_(Tensor & self, Scalar threshold, Scalar value) const {
    runtime_error("threshold_forward_ is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    runtime_error("adaptive_max_pool2d_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool2d(const Tensor & self, IntList output_size) const {
    runtime_error("adaptive_max_pool2d is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) const {
    runtime_error("adaptive_max_pool2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) const {
    runtime_error("adaptive_max_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    runtime_error("adaptive_max_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) const {
    runtime_error("adaptive_max_pool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d is not implemented for type %s", toString());
}
Tensor & Type::avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d is not implemented for type %s", toString());
}
Tensor & Type::avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) const {
    runtime_error("avg_pool3d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool2d_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool2d is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool2d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    runtime_error("max_pool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    runtime_error("max_pool2d_backward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool3d_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool3d is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool3d_forward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) const {
    runtime_error("max_pool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    runtime_error("max_pool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) const {
    runtime_error("max_pool3d_backward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d is not implemented for type %s", toString());
}
Tensor & Type::max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) const {
    runtime_error("max_unpool2d_backward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d is not implemented for type %s", toString());
}
Tensor & Type::max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d_forward is not implemented for type %s", toString());
}
Tensor & Type::max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d_backward_out is not implemented for type %s", toString());
}
Tensor Type::max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) const {
    runtime_error("max_unpool3d_backward is not implemented for type %s", toString());
}
Tensor & Type::_sigmoid_out(Tensor & output, const Tensor & self) const {
    runtime_error("_sigmoid_out is not implemented for type %s", toString());
}
Tensor Type::_sigmoid(const Tensor & self) const {
    runtime_error("_sigmoid is not implemented for type %s", toString());
}
Tensor & Type::_sigmoid_forward_out(Tensor & output, const Tensor & self) const {
    runtime_error("_sigmoid_forward_out is not implemented for type %s", toString());
}
Tensor Type::_sigmoid_forward(const Tensor & self) const {
    runtime_error("_sigmoid_forward is not implemented for type %s", toString());
}
Tensor & Type::_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    runtime_error("_sigmoid_backward_out is not implemented for type %s", toString());
}
Tensor Type::_sigmoid_backward(const Tensor & grad_output, const Tensor & output) const {
    runtime_error("_sigmoid_backward is not implemented for type %s", toString());
}
Tensor & Type::_tanh_out(Tensor & output, const Tensor & self) const {
    runtime_error("_tanh_out is not implemented for type %s", toString());
}
Tensor Type::_tanh(const Tensor & self) const {
    runtime_error("_tanh is not implemented for type %s", toString());
}
Tensor & Type::_tanh_forward_out(Tensor & output, const Tensor & self) const {
    runtime_error("_tanh_forward_out is not implemented for type %s", toString());
}
Tensor Type::_tanh_forward(const Tensor & self) const {
    runtime_error("_tanh_forward is not implemented for type %s", toString());
}
Tensor & Type::_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) const {
    runtime_error("_tanh_backward_out is not implemented for type %s", toString());
}
Tensor Type::_tanh_backward(const Tensor & grad_output, const Tensor & output) const {
    runtime_error("_tanh_backward is not implemented for type %s", toString());
}
Tensor & Type::batch_norm_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    runtime_error("batch_norm_out is not implemented for type %s", toString());
}
Tensor Type::batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) const {
    runtime_error("batch_norm is not implemented for type %s", toString());
}
Tensor & Type::batch_norm_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    runtime_error("batch_norm_forward_out is not implemented for type %s", toString());
}
Tensor Type::batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    runtime_error("batch_norm_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) const {
    runtime_error("batch_norm_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) const {
    runtime_error("batch_norm_backward is not implemented for type %s", toString());
}
Tensor & Type::conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    runtime_error("conv_transpose2d_out is not implemented for type %s", toString());
}
Tensor Type::conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    runtime_error("conv_transpose2d is not implemented for type %s", toString());
}
Tensor & Type::conv_transpose2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_transpose2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_transpose2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_transpose2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    runtime_error("conv_transpose2d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    runtime_error("conv_transpose3d_out is not implemented for type %s", toString());
}
Tensor Type::conv_transpose3d(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) const {
    runtime_error("conv_transpose3d is not implemented for type %s", toString());
}
Tensor & Type::conv_transpose3d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv_transpose3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv_transpose3d_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv_transpose3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv_transpose3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    runtime_error("conv_transpose3d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    runtime_error("conv2d_out is not implemented for type %s", toString());
}
Tensor Type::conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    runtime_error("conv2d is not implemented for type %s", toString());
}
Tensor & Type::conv2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    runtime_error("conv2d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_depthwise2d_out is not implemented for type %s", toString());
}
Tensor Type::conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_depthwise2d is not implemented for type %s", toString());
}
Tensor & Type::conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_depthwise2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_depthwise2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &> Type::conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_depthwise2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) const {
    runtime_error("conv_depthwise2d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    runtime_error("conv3d_out is not implemented for type %s", toString());
}
Tensor Type::conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) const {
    runtime_error("conv3d is not implemented for type %s", toString());
}
Tensor & Type::conv3d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) const {
    runtime_error("conv3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) const {
    runtime_error("conv3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) const {
    runtime_error("conv3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) const {
    runtime_error("conv3d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_dilated2d_out is not implemented for type %s", toString());
}
Tensor Type::conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_dilated2d is not implemented for type %s", toString());
}
Tensor & Type::conv_dilated2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated2d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated2d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated2d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    runtime_error("conv_dilated2d_backward is not implemented for type %s", toString());
}
Tensor & Type::conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_dilated3d_out is not implemented for type %s", toString());
}
Tensor Type::conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) const {
    runtime_error("conv_dilated3d is not implemented for type %s", toString());
}
Tensor & Type::conv_dilated3d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated3d_forward_out is not implemented for type %s", toString());
}
Tensor Type::conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated3d_forward is not implemented for type %s", toString());
}
std::tuple<Tensor &,Tensor &,Tensor &> Type::conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) const {
    runtime_error("conv_dilated3d_backward_out is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor,Tensor> Type::conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) const {
    runtime_error("conv_dilated3d_backward is not implemented for type %s", toString());
}
Tensor Type::type_as(const Tensor & self, const Tensor & other) const {
    return  at::native::type_as(self, other);
}
Tensor Type::expand_as(const Tensor & self, const Tensor & other) const {
    return  at::native::expand_as(self, other);
}
std::vector<Tensor> Type::split(const Tensor & self, int64_t split_size, int64_t dim) const {
    return  at::native::split(self, split_size, dim);
}
std::vector<Tensor> Type::chunk(const Tensor & self, int64_t chunks, int64_t dim) const {
    return  at::native::chunk(self, chunks, dim);
}
int64_t Type::size(const Tensor & self, int64_t dim) const {
    return  at::native::size(self, dim);
}
int64_t Type::stride(const Tensor & self, int64_t dim) const {
    return  at::native::stride(self, dim);
}
Tensor Type::index(const Tensor & self, TensorList indices) const {
    return  at::native::index(self, indices);
}
Tensor & Type::index_put_(Tensor & self, TensorList indices, const Tensor & values) const {
    return  at::native::index_put_(self, indices, values);
}
bool Type::is_same_size(const Tensor & self, const Tensor & other) const {
    return  at::native::is_same_size(self, other);
}
bool Type::is_cuda(const Tensor & self) const {
    return  at::native::is_cuda(self);
}
Tensor Type::select(const Tensor & self, int64_t dim, int64_t index) const {
    return  at::native::select(self, dim, index);
}
Tensor Type::narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) const {
    return  at::native::narrow(self, dim, start, length);
}
Tensor Type::slice(const Tensor & self, int64_t start, int64_t end, int64_t step, int64_t dim) const {
    return  at::native::slice(self, start, end, step, dim);
}
Tensor Type::permute(const Tensor & self, IntList dims) const {
    return  at::native::permute(self, dims);
}
Tensor Type::expand(const Tensor & self, IntList size) const {
    return  at::native::expand(self, size);
}
Tensor Type::squeeze(const Tensor & self) const {
    return  at::native::squeeze(self);
}
Tensor Type::squeeze(const Tensor & self, int64_t dim) const {
    return  at::native::squeeze(self, dim);
}
Tensor & Type::squeeze_(Tensor & self) const {
    return  at::native::squeeze_(self);
}
Tensor & Type::squeeze_(Tensor & self, int64_t dim) const {
    return  at::native::squeeze_(self, dim);
}
Tensor Type::unsqueeze(const Tensor & self, int64_t dim) const {
    return  at::native::unsqueeze(self, dim);
}
Tensor & Type::unsqueeze_(Tensor & self, int64_t dim) const {
    return  at::native::unsqueeze_(self, dim);
}
Tensor Type::stack(TensorList tensors, int64_t dim) const {
    return  at::native::stack(tensors, dim);
}
bool Type::is_signed(const Tensor & self) const {
    runtime_error("is_signed is not implemented for type %s", toString());
}
std::tuple<Tensor,Tensor> Type::RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) const {
    runtime_error("RoiPooling2d_forward is not implemented for type %s", toString());
}
Tensor Type::RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) const {
    runtime_error("RoiPooling2d_backward is not implemented for type %s", toString());
}

}
