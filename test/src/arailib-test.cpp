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
#include "nndescent.hpp"

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
    const std::vector<double> v1{1, 1};
    const std::vector<double> v2{5, 4};
    const double actual = arailib::l2_norm(v1, v2);

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

TEST(nn_descent, neighbor) {
    std::string data_path = "../../../test/data/series.csv";
    arailib::Series series = arailib::read_csv(data_path);

    nndescent::Neighbor actual(series[0], series[1]);
    ASSERT_EQ(actual.point.id, 1);
    ASSERT_EQ(actual.distance, l2_norm(series[0], series[1]));
}

TEST(nn_descent, KNearestNeighbors) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Series series = arailib::read_csv(data_path);

    auto query = series[0];
    auto point_1 = series[1], point_2 = series[2];
    nndescent::KNearestNeighborsHeap knn(5, query);
    knn.update(point_2);
    knn.update(point_1);

    ASSERT_EQ(point_2.id, knn.furthest().point.id);
    knn.pop();
    ASSERT_EQ(point_1.id, knn.furthest().point.id);
}

TEST(nn_descent, sample) {
    Series series = {
        Point(0, {0}),
        Point(1, {3}),
        Point(2, {5}),
        Point(3, {2}),
        Point(4, {4}),
        Point(5, {8}),
        Point(6, {7})
    };

    auto sampled_knn = nndescent::sample(series, series[0], 4).vectorize();

    ASSERT_EQ(sampled_knn[0].point.id, 3);
    ASSERT_EQ(sampled_knn[1].point.id, 4);
    ASSERT_EQ(sampled_knn[2].point.id, 2);
    ASSERT_EQ(sampled_knn[3].point.id, 6);
}

TEST(nn_descent, reverse) {
    Series series = {
        Point(0, {0}),
        Point(1, {3}),
        Point(2, {5}),
        Point(3, {2}),
    };

    size_t k = 2;
    nndescent::KNearestNeighborsHeapList knn_list = {
        nndescent::KNearestNeighborsHeap(k, series[0]),
        nndescent::KNearestNeighborsHeap(k, series[1]),
        nndescent::KNearestNeighborsHeap(k, series[2]),
        nndescent::KNearestNeighborsHeap(k, series[3]),
    };

    knn_list[0].update(series[1]);
    knn_list[0].update(series[3]);

    knn_list[1].update(series[2]);
    knn_list[1].update(series[3]);

    knn_list[2].update(series[1]);
    knn_list[2].update(series[3]);

    knn_list[3].update(series[0]);
    knn_list[3].update(series[1]);

    auto reverse_knn_list = nndescent::reverse(knn_list);

    ASSERT_EQ(reverse_knn_list[0][0].id, 3);

    ASSERT_EQ(reverse_knn_list[1][0].id, 0);
    ASSERT_EQ(reverse_knn_list[1][1].id, 2);
    ASSERT_EQ(reverse_knn_list[1][2].id, 3);

    ASSERT_EQ(reverse_knn_list[2][0].id, 1);

    ASSERT_EQ(reverse_knn_list[3][0].id, 0);
    ASSERT_EQ(reverse_knn_list[3][1].id, 1);
    ASSERT_EQ(reverse_knn_list[3][2].id, 2);
}