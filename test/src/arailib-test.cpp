//
// Created by Yusuke Arai on 2019-08-05.
//
#include <vector>
#include <functional>
#include <algorithm>
#include <queue>
#include <functional>
#include "gtest/gtest.h"
#include "arailib.hpp"

using namespace arailib;

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
    const auto p1 = Point({0, {1, 1}});
    const auto p2 = Point({1, {5, 4}});
    const double actual = arailib::euclidean_distance(p1, p2);

    const double expect = 5;
    ASSERT_EQ(actual, expect);
}

TEST(Object, method_test) {
    const arailib::Point o1(0, {1, 2, 3});
    ASSERT_EQ(o1.id, 0);
    ASSERT_EQ(o1[1], 2);

    arailib::Point o2(0, {1, 2, 3});
    ASSERT_TRUE(o1 == o2);

    o2.id++;
    ASSERT_FALSE(o1 == o2);
}

TEST(Series, read_csv_test) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Series series = arailib::read_csv(data_path);
    ASSERT_EQ(series.size(), 3);

    const arailib::Point actual = series[1];
    arailib::Point expect(1, {2, 3, 4});
    ASSERT_EQ(actual, expect);
}

TEST(Series, write_csv_test) {
    const std::string data_path = "../../../test/data/write_series.csv";
    arailib::Series write_series;
    write_series.push_back(arailib::Point(0, {1, 2, 3}));
    write_series.push_back(arailib::Point(1, {2, 3, 4}));
    write_series.push_back(arailib::Point(2, {3, 4, 5}));
    arailib::write_csv(write_series, data_path);

    const arailib::Series read_series = arailib::read_csv(data_path);
    ASSERT_EQ(read_series.size(), 3);

    const arailib::Point actual = read_series[1];
    arailib::Point expect(1, {2, 3, 4});
    ASSERT_EQ(actual, expect);
}

TEST(Distance, cosine_similarity) {
    const auto p1 = Point(0, {2, 0});
    const auto p2 = Point(1, {2, 2 * static_cast<float>(sqrt(3))});
    const auto actual = cosine_similarity(p1, p2);
    const float expect = 0.5;
    ASSERT_EQ(actual, expect);
}

TEST(Distance, angular_distance) {
    const auto p1 = Point(0, {2, 0});
    const auto p2 = Point(1, {2, 2 * static_cast<float>(sqrt(3))});
    const auto actual = angular_distance(p1, p2);
    const float expect = static_cast<float>(1) / 3;
    ASSERT_EQ(static_cast<int>(actual * 100000), static_cast<int>(expect * 100000));
}