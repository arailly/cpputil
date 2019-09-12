//
// Created by Yusuke Arai on 2019/09/12.
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

TEST(nn_descent, neighbor) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Series series = arailib::read_csv(data_path);

    nndescent::Neighbor actual(series[0], series[1]);
    nndescent::Neighbor expect(1, l2_norm(series[0], series[1]));
    ASSERT_EQ(actual.id, expect.id);
    ASSERT_EQ(actual.distance, expect.distance);
}

TEST(nn_descent, KNearestNeighbors) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Series series = arailib::read_csv(data_path);

    auto query = series[0];
    auto point_1 = series[1], point_2 = series[2];
    nndescent::KNearestNeighbors knn(5, query);
    knn.update(point_2);
    knn.update(point_1);

    ASSERT_EQ(point_2.id, knn.furthest().id);
    knn.pop();
    ASSERT_EQ(point_1.id, knn.furthest().id);
}