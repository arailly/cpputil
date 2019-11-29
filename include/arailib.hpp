#ifndef ARAILIB_ARAILIB_HPP
#define ARAILIB_ARAILIB_HPP

#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <omp.h>
#include "nlohmann/json.hpp"

using namespace std;
using namespace nlohmann;

namespace arailib {
    template<class UnaryOperation, class Iterable>
    Iterable fmap(UnaryOperation op, const Iterable &v) {
        Iterable result;
        std::transform(v.begin(), v.end(), std::back_inserter(result), op);
        return result;
    }

    template<class Predicate, class Iterable>
    Iterable filter(Predicate pred, const Iterable &v) {
        Iterable result;
        std::copy_if(v.begin(), v.end(), std::back_inserter(result), pred);
        return result;
    }

    struct Point {
        size_t id;
        std::vector<float> x;

        Point() : id(0), x({0}) {}

        Point(size_t i, std::vector<float> v) {
            id = i;
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        Point(std::vector<float> v) {
            id = 0;
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        auto& operator [] (size_t i) { return x[i]; }
        const auto& operator [] (size_t i) const { return x[i]; }

        bool operator==(const Point &o) const {
            if (id == o.id) return true;
            return false;
        }

        bool operator!=(const Point &o) const {
            if (id != o.id) return true;
            return false;
        }

        size_t size() const { return x.size(); }
        auto begin() const { return x.begin(); }
        auto end() const { return x.end(); }

        void show() const {
            std::cout << id << ": ";
            for (const auto &xi : x) {
                std::cout << xi << ' ';
            }
            std::cout << std::endl;
        }
    };

    typedef vector<Point> Series;

    class SeriesList {
    private:
        vector<Series> series_list;

    public:
        SeriesList(size_t vector_size) { series_list.resize(vector_size); }

        auto size() const { return series_list.size(); }

        auto begin() const { return series_list.begin(); }

        auto end() const { return series_list.end(); }

        auto &operator[](size_t i) { return series_list[i]; }

        auto &operator[](const Point &p) { return series_list[p.id]; }

        const auto &operator[](size_t i) const { return series_list[i]; }

        const auto &operator[](const Point &p) const { return series_list[p.id]; }
    };

    typedef function<float(Point, Point)> DistanceFunction;

    float euclidean_distance(const Point& p1, const Point& p2) {
        float result = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            result += std::pow(p1[i] - p2[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

    float l2_norm(const Point& p) {
        float result = 0;
        for (size_t i = 0; i < p.size(); i++) {
            result += std::pow(p[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

    float cosine_similarity(const Point& p1, const Point& p2) {
        return static_cast<float>(inner_product(p1.begin(), p1.end(), p2.begin(), 0.0)
            / (l2_norm(p1) * l2_norm(p2)));
    }

    const float pi = static_cast<const float>(3.14159265358979323846264338);
    float angular_distance(const Point& p1, const Point& p2) {
        return acos(cosine_similarity(p1, p2)) / pi;
    }

    template <class T = float>
    vector<T> split(string &input, char delimiter = ',') {
        std::istringstream stream(input);
        std::string field;
        std::vector<T> result;

        while (std::getline(stream, field, delimiter)) {
            result.push_back(std::stod(field));
        }

        return result;
    }

    template <class T = float>
    Series read_csv(const std::string &path, const int& nrows = -1,
                    const bool &skip_header = false) {
        std::ifstream ifs(path);
        if (!ifs) throw runtime_error("Can't open file!");
        std::string line;

        Series series;
        for (size_t i = 0; (i < nrows) && std::getline(ifs, line); ++i) {
            // if first line is the header then skip
            if (skip_header && (i == 0)) continue;
            std::vector<T> v = split<T>(line);
            series.push_back(Point(i, v));
        }
        return series;
    }

    const int n_max_threads = omp_get_max_threads();

    Series load_data(const string& path, int nk = 0) {
        // file path
        if (path.rfind(".csv", path.size()) < path.size()) {
            auto series = Series();
            ifstream ifs(path);
            if (!ifs) throw runtime_error("Can't open file!");
            string line;
            while(getline(ifs, line)) {
                auto v = split(line);
                const auto id = static_cast<size_t>(v[0]);
                v.erase(v.begin());
                series.push_back(Point(id, v));
            }
            return series;
        }

        // dir path
        auto series = Series(nk * 1000);
#pragma omp parallel for
        for (int i = 0; i < nk; i++) {
            const string data_path = path + '/' + to_string(i) + ".csv";
            ifstream ifs(data_path);
            if (!ifs) throw runtime_error("Can't open file!");
            string line;
            while(getline(ifs, line)) {
                auto v = split(line);
                const auto id = static_cast<size_t>(v[0]);
                v.erase(v.begin());
                series[id] = Point(id, v);
            }
        }
        return series;
    }

    template<typename T>
    void write_csv(const std::vector<T> &v, const std::string &path) {
        std::ofstream ofs(path);
        for (const auto &o : v) {
            std::string line;
            for (const auto &e : o) {
                line += std::to_string(e) + ',';
            }
            line.pop_back();
            line += '\n';
            ofs << line;
        }
    }

    json read_config(const string& config_path = "./config.json") {
        json config;
        ifstream ifs(config_path);
        if (ifs.fail()) throw runtime_error("Can't open file!");
        ifs >> config;
        return config;
    }

    struct Edge {
        size_t query_id, point_id;
        Edge(size_t query_id, size_t point_id) : query_id(query_id), point_id(point_id) {};
        Edge(vector<size_t> v) : query_id(v[0]), point_id(v[1]) {};
    };

    typedef vector<Edge> EdgeSeries;
    typedef vector<EdgeSeries> EdgeSeriesList;

    struct Node {
        const Point point;
        vector<reference_wrapper<const Node>> neighbors;
        unordered_map<size_t, bool> added;

        void init() { added[point.id] = true; }
        Node() : point(Point(0, {0})) { init(); }
        Node(Point& p) : point(move(p)) { init(); }

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

    bool is_csv(const string& path) {
        return (path.rfind(".csv", path.size()) < path.size());
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

    // nsw's knn_search
    vector<reference_wrapper<const Node>>
    knn_search(const Point& query, const uint k, const Node& start_node) {
        unordered_map<size_t, bool> checked;
        checked[start_node.point.id] = true;

        multimap<float, reference_wrapper<const Node>> candidates, result_map;
        const auto distance_to_start_node = euclidean_distance(query, start_node.point);
        candidates.emplace(distance_to_start_node, start_node);
        result_map.emplace(distance_to_start_node, start_node);

        while (!candidates.empty()) {
            const auto nearest_pair = candidates.extract(candidates.begin());
            const auto& distance_to_nearest = nearest_pair.key();
            const Node& nearest = nearest_pair.mapped().get();

            auto& furthest = *(--result_map.cend());
            // check if all elements are evaluated
            if (distance_to_nearest > furthest.first) break;

            for (auto& neighbor : nearest.neighbors) {
                if (checked[neighbor.get().point.id]) continue;
                checked[neighbor.get().point.id] = true;

                const auto& distance_to_neighbor = euclidean_distance(query, neighbor.get().point);

                if (result_map.size() < k) {
                    candidates.emplace(distance_to_neighbor, neighbor.get());
                    result_map.emplace(distance_to_neighbor, neighbor.get());
                    continue;
                }

                const auto& furthest_ = *(--result_map.end());
                const auto& distance_to_furthest_ = euclidean_distance(query, furthest_.second.get().point);

                if (distance_to_neighbor < distance_to_furthest_) {
                    candidates.emplace(distance_to_neighbor, neighbor.get());
                    result_map.emplace(distance_to_neighbor, neighbor.get());
                    result_map.erase(--result_map.cend());
                }
            }
        }
        // result_map => result vector;
        return [&result_map]() {
            vector<reference_wrapper<const Node>> r;
            for (const auto& neighbor : result_map) r.push_back(neighbor.second.get());
            return r;
        }();
    }

    // nsg's knn_search
    vector<reference_wrapper<const Node>>
    knn_search(const Point query, const unsigned k,
               const Node& start_node, const unsigned n_candidate = 40) {
        const auto start = chrono::system_clock::now();

        unordered_map<size_t, bool> checked, added;
        added[start_node.point.id] = true;

        multimap<float, reference_wrapper<const Node>> candidates;
        const auto distance_to_start_node = euclidean_distance(query, start_node.point);
        candidates.emplace(distance_to_start_node, start_node);

        while (true) {
            bool is_updated = false;
            for (auto candidate_pair_ptr = candidates.begin();
                 candidate_pair_ptr != candidates.end();
                 candidate_pair_ptr++) {

                const auto& candidate = candidate_pair_ptr->second.get();
                if (checked[candidate.point.id]) continue;
                checked[candidate.point.id] = true;
                is_updated = true;

                for (const auto& neighbor : candidate.neighbors) {
                    if (added[neighbor.get().point.id]) continue;
                    added[neighbor.get().point.id] = true;

                    const auto d = euclidean_distance(query, neighbor.get().point);
                    candidates.emplace(d, neighbor.get());
                }
                // resize candidates n_candidate
                while (candidates.size() > n_candidate) candidates.erase(--candidates.cend());
                candidate_pair_ptr = candidates.begin();
            }
            if (!is_updated) break;
        }

        vector<reference_wrapper<const Node>> result;
        for (const auto& c : candidates) {
            result.emplace_back(c.second.get());
            if (result.size() >= k) break;
        }
        return result;
    }
}

#endif //ARAILIB_ARAILIB_HPP