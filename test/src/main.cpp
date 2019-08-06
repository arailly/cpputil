//
// Created by Yusuke Arai on 2019-08-05.
//
#include <vector>
#include <functional>
#include <algorithm>
#include "gtest/gtest.h"
#include "arailib.hpp"

TEST(Functional, fmap_test) {
    std::vector<int> v{1, 2, 3};
    const auto double_func = [](int x) { return x * 3; };
    const auto actual = fmap(double_func, v);

    const std::vector<int> expect{3, 6, 9};
    EXPECT_EQ(actual, expect);
}

TEST(Functional, filter_test) {
    std::vector<int> v{1, 2, 3, 4, 5};
    const auto pred = [](int x) { return (x > 3); };
    const auto actual = filter(pred, v);

    const std::vector<int> expect{4, 5};
    EXPECT_EQ(actual, expect);
}