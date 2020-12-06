/**
 * @brief
 * ragged_utils_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu, Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/context.h"
#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/test_utils.h"
#include "k2/csrc/tensor.h"
#include "k2/csrc/ragged_utils.h"

namespace k2 {

TEST(RaggedUtilsTest, CheckAxisEqual) {

  RaggedShape shape1(" [[ x x x ] [ x x ]]"),
      shape1b(" [[ x x x ] [ x x ]]"),
      shape2("[[ x x x ] [ x ]]");

  RaggedShape *array[] = { &shape1, &shape1b, &shape1, &shape2, &shape2 };
  int32_t axis = 0;
  CheckAxisEqual(0, axis, array);
  CheckAxisEqual(1, axis, array);
  CheckAxisEqual(2, axis, array);
  CheckAxisEqual(3, axis, array);
#ifndef NDEBUG
  // this won't actualy die if we compiled with NDEBUG.
  ASSERT_DEATH(CheckAxisEqual(4, axis, array), "");
#endif
  CheckAxisEqual(2, axis, array + 3);
}

TEST(RaggedUtilsTest, GetLayer) {

  RaggedShape shape1(" [[[ x x x ] [ x x ]]]"),
      shape2(" [[ x x x ] [ x x ]]"),
      shape3("[[x x]]");

  RaggedShape shape2b = GetLayer(shape1, 1),
              shape3b = GetLayer(shape1, 0);
  ASSERT_TRUE(Equal(shape2, shape2b));
  ASSERT_TRUE(Equal(shape3, shape3b));
}


TEST(RaggedUtilsTest, IntersperseRaggedLayerSimpleLayer0) {
  RaggedShape shape1(" [[ x x x ] [ x x ]]"),
      shape2(" [[ x ] [ ]]"),
      shape3("[[x x x] [ x] [ x x ] [ ]]");

  RaggedShape *shapes[] = {&shape1, &shape2};
  int32_t layer = 0;
  Array1<uint32_t> merge_map;
  RaggedShape shape = IntersperseRaggedLayer(2, layer, shapes, &merge_map);
  std::vector<uint32_t> merge_values = { 0, 2, 4, 1, 6, 8 };
  CheckArrayData(merge_map, merge_values);
  ASSERT_TRUE(Equal(shape, shape3));
}

TEST(RaggedUtilsTest, IntersperseRaggedLayerSimpleLayer1) {
  RaggedShape shape1(" [[[ x x x ] [ x x ]]]"),
      shape2(" [[[ x ] [ ]]]"),
      shape3("[[x x x] [ x] [ x x ] [ ]]");

  RaggedShape *shapes[] = {&shape1, &shape2};
  int32_t layer = 1;
  RaggedShape shape = IntersperseRaggedLayer(2, layer, shapes, nullptr);
  ASSERT_TRUE(Equal(shape, shape3));
}


TEST(RaggedUtilsTest, IntersperseRaggedLayerLong) {
  RaggedShape shape1(" [[ x x x ] [ x x ]]"),
      shape2(" [[ x ] [ ]]");

  RaggedShape *shapes[100];

  for (int32_t i = 0; i < 20; i++) {
    shapes[i] = i % 2 == 0 ? &shape1 : &shape2;
  }
  int32_t layer = 0;
  Array1<uint32_t> merge_map;

  RaggedShape shape = IntersperseRaggedLayer(20, layer, shapes, &merge_map);
  K2_LOG(INFO) << "merge_map = " << merge_map;

  ContextPtr cpu = GetCpuContext();
  shape = shape.To(cpu);
  shape1 = shape1.To(cpu);
  shape2 = shape2.To(cpu);
  merge_map = merge_map.To(cpu);
  K2_CHECK_EQ(shape.Dim0(), 20 * 2);
  int32_t *shape_row_splits1_data = shape.RowSplits(1).Data();
  for (int32_t i = 0; i < (20 * 2); i++) {  // to because each shape1,shape2 have 2 sub-lists.
    RaggedShape &src_shape = (i % 2 == 0 ? shape1 : shape2);
    int32_t *src_shape_row_splits1_data = src_shape.RowSplits(1).Data();
    int32_t row_begin = shape_row_splits1_data[i],
              row_end = shape_row_splits1_data[i+1];
    int32_t src_shape_index = (i < 20 ? 0 : 1),
              src_row_begin = src_shape_row_splits1_data[src_shape_index],
                src_row_end = src_shape_row_splits1_data[src_shape_index + 1];
    K2_CHECK_EQ(row_end - row_begin, src_row_end - src_row_begin);

    for (int32_t idx = row_begin; idx < row_end; idx++) {
      int32_t merge_map_value = merge_map[idx],
                      src_idx = merge_map_value % 20,  // 20 == num_srcs
                      src_pos = merge_map_value / 20;
      K2_CHECK_EQ(src_idx, i % 20);
      K2_CHECK_EQ(idx - row_begin, src_pos - src_row_begin);
    }
  }
}




}  // namespace k2
