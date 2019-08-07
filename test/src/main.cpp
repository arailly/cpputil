//
// Created by Yusuke Arai on 2019-08-05.
//
#include <vector>
#include <functional>
#include <algorithm>
#include "gtest/gtest.h"
#include "arailib.hpp"

#include <unistd.h>

TEST(Functional, fmap_test) {
    std::vector<int> v{1, 2, 3};
    const auto double_func = [](int x) { return x * 3; };
    const auto actual = arailib::fmap(double_func, v);

    const std::vector<int> expect{3, 6, 9};
    ASSERT_EQ(actual, expect);
}

TEST(Functional, filter_test) {
    std::vector<int> v{1, 2, 3, 4, 5};
    const auto pred = [](int x) { return (x > 3); };
    const auto actual = arailib::filter(pred, v);

    const std::vector<int> expect{4, 5};
    ASSERT_EQ(actual, expect);
}

TEST(Utilities, l2_norm_test) {
    const std::vector<double> v1{1, 1};
    const std::vector<double> v2{5, 4};
    const double actual = arailib::l2_norm(v1, v2);

    const double expect = 5;
    ASSERT_EQ(actual, expect);
}

TEST(Object, method_test) {
    const arailib::Object o1(0, {1, 2, 3});
    ASSERT_EQ(o1.index, 0);
    ASSERT_EQ(o1[1], 2);

    arailib::Object o2(0, {1, 2, 3});
    ASSERT_TRUE(o1 == o2);

    o2.index++;
    ASSERT_FALSE(o1 == o2);
}

TEST(Series, read_csv_test) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Series series = arailib::read_csv(data_path);
    const arailib::Object actual = series[1];

    arailib::Object expect(1, {2, 3, 4});
    ASSERT_EQ(actual, expect);
}