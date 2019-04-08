// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include <cstdlib>
#include <sstream>
#include <cstdlib>

namespace onnxruntime {

void* CPUAllocator::Alloc(size_t size) {
  if (size <= 0)
    return nullptr;
  //default align to 64;
  void* p;
#ifdef _WIN32
  size_t alignment = 32;
#else  
  size_t alignment = 64;
#endif
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr) throw std::bad_alloc();
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr) throw std::bad_alloc();
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0) throw std::bad_alloc();
#endif
  return p;
}

void CPUAllocator::Free(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}

const OrtAllocatorInfo& CPUAllocator::Info() const {
  return *allocator_info_;
}
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const OrtAllocatorInfo& info) {
  return (out << info.ToString());
}

ORT_API_STATUS_IMPL(OrtCreateAllocatorInfo, _In_ const char* name1, OrtAllocatorType type, int id1,
                    OrtMemType mem_type1, _Out_ OrtAllocatorInfo** out) {
  *out = new OrtAllocatorInfo(name1, type, id1, mem_type1);
  return nullptr;
}

ORT_API(void, OrtReleaseAllocatorInfo, _Frees_ptr_opt_ OrtAllocatorInfo* p) {
  delete p;
}

ORT_API(const char*, OrtAllocatorInfoGetName, _In_ OrtAllocatorInfo* ptr) {
  return ptr->name;
}

ORT_API(int, OrtAllocatorInfoGetId, _In_ OrtAllocatorInfo* ptr) {
  return ptr->id;
}

ORT_API(OrtMemType, OrtAllocatorInfoGetMemType, _In_ OrtAllocatorInfo* ptr) {
  return ptr->mem_type;
}

ORT_API(OrtAllocatorType, OrtAllocatorInfoGetType, _In_ OrtAllocatorInfo* ptr) {
  return ptr->type;
}

ORT_API(int, OrtCompareAllocatorInfo, _In_ const OrtAllocatorInfo* info1, _In_ const OrtAllocatorInfo* info2) {
  if (*info1 == *info2) {
    return 0;
  }
  return -1;
}
