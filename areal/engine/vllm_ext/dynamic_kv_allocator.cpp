// SPDX-License-Identifier: Apache-2.0
// Reserve address space first; commit individual Ascend physical pages on demand.
#include <acl/acl.h>
#include <map>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>

namespace {
struct Region {
  size_t size;
  int device;
  std::map<uintptr_t, aclrtDrvMemHandle> pages;
};
std::map<uintptr_t, Region> regions;
std::mutex mutex;
thread_local int last_error = 0;
size_t granularity(int device) {
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  size_t value = 0;
  last_error =
      aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, &value);
  return last_error ? 0 : value;
}
auto owner(uintptr_t address) {
  auto it = regions.upper_bound(address);
  if (it == regions.begin()) return regions.end();
  --it;
  return address < it->first + it->second.size ? it : regions.end();
}
}  // namespace

extern "C" {
size_t dkv_granularity(int device) { return granularity(device); }
int dkv_error() { return last_error; }
int dkv_copy(uintptr_t dst, uintptr_t src, size_t size, int kind) {
  if (kind != ACL_MEMCPY_HOST_TO_DEVICE && kind != ACL_MEMCPY_DEVICE_TO_HOST) return -1;
  return aclrtMemcpy(reinterpret_cast<void *>(dst), size, reinterpret_cast<void *>(src), size,
                     static_cast<aclrtMemcpyKind>(kind));
}

void *dkv_malloc(ssize_t size, int device, aclrtStream) {
  std::lock_guard<std::mutex> lock(mutex);
  size_t page = granularity(device);
  if (!page || size <= 0) return nullptr;
  size_t aligned = (size + page - 1) / page * page;
  void *ptr = nullptr;
  last_error = aclrtReserveMemAddress(&ptr, aligned, 0, nullptr, 0);
  if (last_error) {
    std::fprintf(stderr, "dynamic KV reserve failed: %d\n", last_error);
    return nullptr;
  }
  regions.emplace(reinterpret_cast<uintptr_t>(ptr), Region{aligned, device, {}});
  return ptr;
}

int dkv_map(uintptr_t address, size_t size) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = owner(address);
  if (it == regions.end() || address + size > it->first + it->second.size) return -1;
  auto &region = it->second;
  size_t page = granularity(region.device);
  if (!page || size != page || address % page) return -2;
  if (region.pages.count(address)) return 0;
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = region.device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  aclrtDrvMemHandle handle;
  int rc = aclrtMallocPhysical(&handle, size, &prop, 0);
  if (rc) return rc;
  rc = aclrtMapMem(reinterpret_cast<void *>(address), size, 0, handle, 0);
  if (rc) {
    aclrtFreePhysical(handle);
    return rc;
  }
  region.pages.emplace(address, handle);
  // Newly committed KV pages must not expose stale NaNs or padding contents.
  return aclrtMemset(reinterpret_cast<void *>(address), size, 0, size);
}

int dkv_unmap(uintptr_t address) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = owner(address);
  if (it == regions.end()) return -1;
  auto &pages = it->second.pages;
  auto page = pages.find(address);
  if (page == pages.end()) return 0;
  int rc = aclrtUnmapMem(reinterpret_cast<void *>(address));
  if (rc) return rc;
  rc = aclrtFreePhysical(page->second);
  if (rc) return rc;
  pages.erase(page);
  return 0;
}

void dkv_free(void *ptr, ssize_t, int, aclrtStream) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = regions.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == regions.end()) return;
  // PyTorch owns the tensor lifetime. Caller must fence before releasing tensors.
  for (auto &page : it->second.pages) {
    int rc = aclrtUnmapMem(reinterpret_cast<void *>(page.first));
    if (!rc) rc = aclrtFreePhysical(page.second);
    if (rc) {
      std::fprintf(stderr, "dynamic KV teardown failed: %d\n", rc);
      return;
    }
  }
  int rc = aclrtReleaseMemAddress(ptr);
  if (rc) {
    std::fprintf(stderr, "dynamic KV address release failed: %d\n", rc);
    return;
  }
  regions.erase(it);
}
}
