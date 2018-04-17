#include "core/framework/bfc_arena.h"
#include "gtest/gtest.h"
#include <cstdlib>

namespace Lotus {
namespace Test {
static void CheckStats(BFCArena* a, int64_t num_allocs, int64_t bytes_in_use,
                       int64_t max_bytes_in_use, int64_t max_alloc_size) {
  AllocatorStats stats;
  a->GetStats(&stats);
  EXPECT_EQ(stats.bytes_in_use, bytes_in_use);
  EXPECT_EQ(stats.max_bytes_in_use, max_bytes_in_use);
  EXPECT_EQ(stats.num_allocs, num_allocs);
  EXPECT_EQ(stats.max_alloc_size, max_alloc_size);
}

TEST(BFCArenaTest, NoDups) {
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  // Allocate a lot of raw pointers
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.Alloc(s);
    ptrs.push_back(raw);
  }
  CheckStats(&a, 1023, 654336, 654336, 1024);

  std::sort(ptrs.begin(), ptrs.end());

  // Make sure none of them are equal, and that none of them overlap.
  for (size_t i = 1; i < ptrs.size(); i++) {
    ASSERT_NE(ptrs[i], ptrs[i - 1]);  // No dups
    size_t req_size = a.RequestedSize(ptrs[i - 1]);
    ASSERT_GT(req_size, 0);
    ASSERT_GE(static_cast<size_t>(static_cast<char*>(ptrs[i]) - static_cast<char*>(ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < ptrs.size(); i++) {
    a.Free(ptrs[i]);
  }
  CheckStats(&a, 1023, 0, 654336, 1024);
}

TEST(BFCArenaTest, AllocationsAndDeallocations) {
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);
  // Allocate 256 raw pointers of sizes between 100 bytes and about a meg
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % 1048576, 100), 1048576);
    void* raw = a.Alloc(size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.Free(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  // Ensure out of memory errors work and do not prevent future allocations from
  // working.
  void* out_of_memory_ptr = a.Alloc((1 << 30) + 1);
  EXPECT_EQ(out_of_memory_ptr, nullptr);

  // Allocate a lot of raw pointers
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % 1048576, 100), 1048576);
    void* raw = a.Alloc(size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    EXPECT_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<size_t>(static_cast<char*>(existing_ptrs[i]) -
                                  static_cast<char*>(existing_ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.Free(existing_ptrs[i]);
  }
}

TEST(BFCArenaTest, ExerciseCoalescing) {
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  void* first_ptr = a.Alloc(4096);
  a.Free(first_ptr);
  CheckStats(&a, 1, 0, 4096, 4096);
  for (int i = 0; i < 1024; ++i) {
    // Allocate several buffers of different sizes, and then clean them
    // all up.  We should be able to repeat this endlessly without
    // causing fragmentation and growth.
    void* t1 = a.Alloc(4096);

    void* t2 = a.Alloc(1048576 * sizeof(int64_t));
    void* t3 = a.Alloc(2048 * sizeof(double));
    void* t4 = a.Alloc(1048576 * sizeof(int64_t) * sizeof(float));

    a.Free(t1);
    a.Free(t2);
    a.Free(t3);
    a.Free(t4);
  }
  CheckStats(&a, 4097, 0,
             1024 * sizeof(float) + 1048576 * sizeof(int64_t) +
                 2048 * sizeof(double) + 1048576 * sizeof(int64_t) * sizeof(float),
             1048576 * sizeof(int64_t) * sizeof(float));

  // At the end, we should have coalesced all memory into one region
  // starting at the beginning, so validate that allocating a pointer
  // starts from this region.
  void* first_ptr_after = a.Alloc(1024 * sizeof(float));
  EXPECT_EQ(first_ptr, first_ptr_after);
  a.Free(first_ptr_after);
}

TEST(BFCArenaTest, AllocateZeroBufSize) {
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);
  void* ptr = a.Alloc(0);
  EXPECT_EQ(nullptr, ptr);
}

TEST(BFCArenaTest, AllocatedVsRequested) {
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);
  void* t1 = a.Alloc(4);
  EXPECT_EQ(4, a.RequestedSize(t1));
  EXPECT_EQ(256, a.AllocatedSize(t1));
  a.Free(t1);
}

TEST(BFCArenaTest, TestCustomMemoryLimit) {
  // Configure a 1MiB byte limit
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 20);

  void* first_ptr = a.Alloc(sizeof(float) * (1 << 6));
  void* second_ptr = a.Alloc(sizeof(float) * (1 << 20));

  EXPECT_NE(nullptr, first_ptr);
  EXPECT_EQ(nullptr, second_ptr);
  a.Free(first_ptr);
}

TEST(BFCArenaTest, AllocationsAndDeallocationsWithGrowth) {
  // Max of 2GiB, but starts out small.
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1LL << 31);

  // Allocate 10 raw pointers of sizes between 100 bytes and about
  // 64 megs.
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  const int32_t max_mem = 1 << 27;

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % max_mem, 100), max_mem);
    void* raw = a.Alloc(size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.Free(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  const int32_t max_mem_2 = 1 << 26;
  // Allocate a lot of raw pointers between 100 bytes and 64 megs.
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand() % max_mem_2, 100), max_mem_2);
    void* raw = a.Alloc(size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    EXPECT_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<size_t>(
                  static_cast<char*>(existing_ptrs[i]) -
                  static_cast<char*>(existing_ptrs[i - 1])),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.Free(existing_ptrs[i]);
  }
}

TEST(BFCArenaTest, TestReserve) {
  // Configure a 1MiB byte limit
  BFCArena a(std::unique_ptr<IDeviceAllocator>(new CPUAllocator()), 1 << 30);

  void* first_ptr = a.Alloc(sizeof(float) * (1 << 6));
  void* second_ptr = a.Reserve(sizeof(float) * (1 << 20));
  a.Free(first_ptr);
  a.Free(second_ptr);

  AllocatorStats stats;
  a.GetStats(&stats);
  EXPECT_EQ(stats.total_allocated_bytes_, 1048576);
}
}  // namespace Test
}  // namespace Lotus
