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
#include <graph.hpp>

using namespace arailib;
using namespace graph;

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

TEST(Series, load_data_dir) {
    const string data_dir = "/Users/yusuke-arai/workspace/dataset/sift/sift_base/";
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    const auto actual = load_data(data_dir, 2);
    const auto expect = read_csv(data_path, 2000);
    ASSERT_EQ(actual.size(), expect.size());
    ASSERT_EQ(actual[1999].id, expect[1999].id);
}

TEST(Series, load_data_file) {
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    const auto actual = load_data(data_path, 2000);
    const auto expect = read_csv(data_path, 2000);
    ASSERT_EQ(actual.size(), expect.size());
    ASSERT_EQ(actual[0].x.size(), expect[0].x.size());
    ASSERT_EQ(actual[1999].id, expect[1999].id);
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

TEST(graph, node) {
    auto p0 = Point(0, {1});
    auto p1 = Point(1, {2});
    auto p2 = Point(2, {3});
    auto p3 = Point(3, {4});
    auto p4 = Point(4, {5});

    auto n0 = Node(p0);
    auto n1 = Node(p1);
    auto n2 = Node(p2);
    auto n3 = Node(p3);
    auto n4 = Node(p4);

    n0.add_neighbor(n1);
    n0.add_neighbor(n2);
    n1.add_neighbor(n3);
    n1.add_neighbor(n4);

    ASSERT_EQ(n0.neighbors[0].get().point.id, 1);
    ASSERT_EQ(n0.neighbors[1].get().point.id, 2);

    ASSERT_EQ(n0.neighbors[0].get().neighbors[0].get().point.id, 3);
    ASSERT_EQ(n0.neighbors[0].get().neighbors[1].get().point.id, 4);
}

TEST(graph, create_from_series) {
    const auto p0 = Point(0, {1});
    const auto p1 = Point(1, {2});
    const auto p2 = Point(2, {3});
    const auto p3 = Point(3, {4});
    const auto p4 = Point(4, {5});

    auto series = Series{p0, p1, p2, p3, p4};
    const auto graph = Graph(series);
    auto a = 1;
}

TEST(graph, create_graph_from_file) {
    string graph_path = "/Users/yusuke-arai/workspace/index/knn-graph/sift_k_5_n_100.csv";
    string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    int n = 100;

    const auto graph = create_graph_from_file(data_path, graph_path, n);
    ASSERT_EQ(graph[0].point.size(), 128);
    ASSERT_EQ(graph[0].neighbors.size(), 5);
    ASSERT_EQ(graph[0].neighbors[0].get().point.id, 2);
    ASSERT_EQ(graph[0].neighbors[1].get().point.id, 6);
}

//TEST(graph, knn_search) {
//    const auto query = Point(5, {6});
//    const auto k = 3;
//
//    const auto p0 = Point(0, {1});
//    const auto p1 = Point(1, {2});
//    const auto p2 = Point(2, {3});
//    const auto p3 = Point(3, {4});
//    const auto p4 = Point(4, {5});
//
//    auto series = Series{p0, p1, p2, p3, p4};
//    auto graph = Graph(series);
//    graph[0].add_neighbor(graph[1]);
//    graph[1].add_neighbor(graph[2]);
//    graph[2].add_neighbor(graph[3]);
//    graph[3].add_neighbor(graph[4]);
//
//    const auto result = knn_search(query, k, graph[0]);
//    ASSERT_EQ(result.size(), k);
//    ASSERT_EQ(result[0].get().point.id, 4);
//    ASSERT_EQ(result[1].get().point.id, 3);
//    ASSERT_EQ(result[2].get().point.id, 2);
//}

TEST(graph, load_graph) {
    unsigned n = 1000;
    const string data_path = "/Users/yusuke-arai/workspace/dataset/sift/sift_base.csv";
    const string graph_path = "/Users/yusuke-arai/workspace/index/nsg-sift1k-m20.csv";
    const auto graph = load_graph(data_path, graph_path, n);
    ASSERT_EQ(graph.size(), n);
    ASSERT_EQ(graph[0].get_n_neighbors(), 34);
    ASSERT_EQ(graph[0].neighbors[0].get().point.id, 2);
}