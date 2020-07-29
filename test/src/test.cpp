//
// Created by Yusuke Arai on 2019-08-05.
//
#include <vector>
#include <functional>
#include <algorithm>
#include <queue>
#include <functional>
#include "gtest/gtest.h"
#include <arailib.hpp>
//#include <graph.hpp>

using namespace arailib;
//using namespace graph;

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
    const auto p1 = Data<>({0, {1, 1}});
    const auto p2 = Data<>({1, {5, 4}});
    const double actual = arailib::euclidean_distance(p1, p2);

    const double expect = 5;
    ASSERT_EQ(actual, expect);
}

TEST(Object, method_test) {
    const arailib::Data<> o1(0, {1, 2, 3});
    ASSERT_EQ(o1.id, 0);
    ASSERT_EQ(o1[1], 2);

    arailib::Data<> o2(0, {1, 2, 3});
    ASSERT_TRUE(o1 == o2);

    o2.id++;
    ASSERT_FALSE(o1 == o2);
}

TEST(Series, read_csv_test) {
    const std::string data_path = "../../../test/data/series.csv";
    const arailib::Dataset<> series = arailib::read_csv(data_path);
    ASSERT_EQ(series.size(), 3);

    const arailib::Data<> actual = series[1];
    arailib::Data<> expect(1, {2, 3, 4});
    ASSERT_EQ(actual, expect);
}

TEST(Series, load_data_dir) {
    const string data_dir = "/home/arai/workspace/dataset/sift/sift_base/";
    const string data_path = "/home/arai/workspace/dataset/sift/sift_base.csv";
    const auto actual = load_data(data_dir, 2);
    const auto expect = read_csv(data_path, 2000);
    ASSERT_EQ(actual.size(), expect.size());
    ASSERT_EQ(actual[1999].id, expect[1999].id);
}

TEST(Series, load_data_file) {
    const string data_path = "/home/arai/workspace/dataset/sift/sift_base.csv";
    const auto actual = load_data(data_path, 2000);
    const auto expect = read_csv(data_path, 2000);
    ASSERT_EQ(actual.size(), expect.size());
    ASSERT_EQ(actual[0].x.size(), expect[0].x.size());
    ASSERT_EQ(actual[1999].id, expect[1999].id);
}

TEST(Series, write_csv_test) {
    const std::string data_path = "../../../test/data/write_series.csv";
    arailib::Dataset<> write_series;
    write_series.push_back(arailib::Data<>(0, {1, 2, 3}));
    write_series.push_back(arailib::Data<>(1, {2, 3, 4}));
    write_series.push_back(arailib::Data<>(2, {3, 4, 5}));
    arailib::write_csv(write_series, data_path);

    const arailib::Dataset<> read_series = arailib::read_csv(data_path);
    ASSERT_EQ(read_series.size(), 3);

    const arailib::Data<> actual = read_series[1];
    arailib::Data<> expect(1, {2, 3, 4});
    ASSERT_EQ(actual, expect);
}

TEST(Distance, cosine_similarity) {
    const auto p1 = Data<>(0, {2, 0});
    const auto p2 = Data<>(1, {2, 2 * static_cast<float>(sqrt(3))});
    const auto actual = cosine_similarity(p1, p2);
    const float expect = 0.5;
    ASSERT_EQ(actual, expect);
}

TEST(Distance, angular_distance) {
    const auto p1 = Data<>(0, {2, 0});
    const auto p2 = Data<>(1, {2, 2 * static_cast<float>(sqrt(3))});
    const auto actual = angular_distance(p1, p2);
    const float expect = static_cast<float>(1) / 3;
    ASSERT_EQ(static_cast<int>(actual * 100000), static_cast<int>(expect * 100000));
}

TEST(Distance, manhattan_distance) {
    auto const p0 = Data<>(0, {2, 0});
    auto const p1 = Data<>(0, {4, -2});
    auto const actual = manhattan_distance(p0, p1);
    auto const expect = 4;
    ASSERT_EQ(actual, expect);
}

TEST(utility, clip) {
    const auto val_1 = 5, val_2 = 50, val_3 = 999;
    const auto min_val = 10;
    const auto max_val = 100;

    ASSERT_EQ(clip(val_1, min_val, max_val), min_val);
    ASSERT_EQ(clip(val_2, min_val, max_val), val_2);
    ASSERT_EQ(clip(val_3, min_val, max_val), max_val);
}

TEST(search, scan_knn_search) {
    Dataset<> series;
    int n = 10;
    for (int i = 0; i < n; ++i) {
        series.push_back(Data<>(i, {(double)i}));
    }

    const Data<> query({10});

    int k = 3;
    const auto result = scan_knn_search(query, k, series);

    ASSERT_EQ(result.size(), k);
    ASSERT_EQ(result[0].id, series[n - 1].id);
    ASSERT_EQ(result[1].id, series[n - 2].id);
}

TEST(util, calc_centroid) {
    Dataset<> dataset;
    dataset.emplace_back(0, vector<double>{1, 2});
    dataset.emplace_back(1, vector<double>{5, 11});

    const auto centroid = calc_centroid(dataset);
    ASSERT_EQ(centroid[0], 3);
    ASSERT_EQ(centroid[1], 6.5);
}

TEST(util, calc_medoid) {
    Dataset<> dataset;
    dataset.emplace_back(0, vector<double>{1, 2});
    dataset.emplace_back(1, vector<double>{5, 5});
    dataset.emplace_back(2, vector<double>{8, 11});

    const auto medoid = calc_medoid(dataset);
    ASSERT_EQ(medoid, 1);
}