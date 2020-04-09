//
// Created by Yusuke Arai on 2019/12/05.
//

#ifndef ARAILIB_GRAPH_HPP
#define ARAILIB_GRAPH_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <omp.h>
#include <arailib.hpp>

using namespace std;
using namespace arailib;

namespace graph {
    struct Edge {
        size_t query_id, point_id;
        Edge(size_t query_id, size_t point_id) : query_id(query_id), point_id(point_id) {};
        Edge(vector<size_t> v) : query_id(v[0]), point_id(v[1]) {};
    };

    typedef vector<Edge> EdgeSeries;
    typedef vector<EdgeSeries> EdgeSeriesList;

    struct Node {
        const Data point;
        vector<reference_wrapper<const Node>> neighbors;
        unordered_map<size_t, bool> added;

        void init() { added[point.id] = true; }
        Node() : point(Data(0, {0})) { init(); }
        Node(Data& p) : point(move(p)) { init(); }

        void add_neighbor(const Node& node) {
            if (added.find(node.point.id) != added.end()) return;
            added[node.point.id] = true;
            neighbors.push_back(node);
        }

        auto get_n_neighbors() const { return neighbors.size(); }
    };

    struct Edge2 {
        Node& first;
        Node& second;

        Edge2(Node& f, Node& s) : first(f), second(s) {}
    };

    typedef vector<Edge2> EdgeList;

    struct Graph {
        vector<Node> nodes;

        Graph(Series& series) {
            for (auto& point : series) {
                nodes.emplace_back(point);
            }
        }

        void set_edge(EdgeList& edge_list) {
            for (auto& edge : edge_list) {
                nodes[edge.first.point.id].add_neighbor(edge.second);
            }
        }

        auto size() const { return nodes.size(); }
        auto begin() const { return nodes.begin(); }
        auto end() const { return nodes.end(); }
        auto& operator [] (size_t i) { return nodes[i]; }
        auto& operator [] (const Node& n) { return nodes[n.point.id]; }
        const auto& operator [] (size_t i) const { return nodes[i]; }
        const auto& operator [] (const Node& n) const { return nodes[n.point.id]; }
    };

    Graph create_graph_from_file(const string& data_path, const string& graph_path, int n = -1) {
        auto series = read_csv(data_path, n);

        ifstream ifs(graph_path);
        if (!ifs) throw runtime_error("Can't open file!");

        Graph graph(series);
        string line;
        while (getline(ifs, line)) {
            auto&& id_pair = split<size_t>(line);
            graph[id_pair[0]].add_neighbor(graph[id_pair[1]]);
        }

        return graph;
    }

    Graph create_graph_from_file(Series& series, const string& graph_path) {
        ifstream ifs(graph_path);
        if (!ifs) throw runtime_error("Can't open file!");

        Graph graph(series);
        string line;
        while (getline(ifs, line)) {
            auto&& id_pair = split<size_t>(line);
            graph[id_pair[0]].add_neighbor(graph[id_pair[1]]);
        }

        return graph;
    }


    Graph load_graph(Series& series, const string& graph_path, int n) {
        // load nsg
        return [&series, &graph_path, n]() {
            // csv
            if (is_csv(graph_path)) {
                ifstream ifs(graph_path);
                if (!ifs) throw runtime_error("Can't open file!");

                Graph graph(series);
                string line;
                while (getline(ifs, line)) {
                    const auto&& ids = split<size_t>(line);
                    graph[ids[0]].add_neighbor(graph[ids[1]]);
                }
                return graph;
            }

            // dir
            Graph graph(series);

#pragma omp parallel for
            for (int i = 0; i < n; i++) {
                const string path = graph_path + "/" + to_string(i) + ".csv";
                ifstream ifs(path);
                string line;
                while (getline(ifs, line)) {
                    const auto ids = split<size_t>(line);
                    graph[ids[0]].add_neighbor(graph[ids[1]]);
                }
            }
            return graph;
        }();
    }

    Graph load_graph(const string& data_path, const string& graph_path, int n) {
        // load data
        auto series = [&data_path, n]() {
            // csv
            if (is_csv(data_path)) return read_csv(data_path, n);
            // dir
            return load_data(data_path, n);
        }();

        // load nsg
        return load_graph(series, graph_path, n);
    }

    void write_graph(const string& save_path, const Graph& graph) {
        // csv
        if (is_csv(save_path)) {
            ofstream ofs(save_path);
            string line;
            for (const auto& node : graph) {
                line = to_string(node.point.id);
                for (const auto& neighbor : node.neighbors) {
                    line += ',' + to_string(neighbor.get().point.id);
                }
                line += '\n';
                ofs << line;
            }
            return;
        }

        // dir
        vector<string> lines(ceil(graph.size() / 1000.0));
        for (const auto& node : graph) {
            const size_t line_i = node.point.id / 1000;
            for (const auto& neighbor : node.neighbors) {
                lines[line_i] += to_string(node.point.id) + ',' +
                                 to_string(neighbor.get().point.id) + '\n';
            }
        }

        for (int i = 0; i < lines.size(); i++) {
            const string path = save_path + "/" + to_string(i) + ".csv";
            ofstream ofs(path);
            ofs << lines[i];
        }
    }
}

#endif //ARAILIB_GRAPH_HPP
