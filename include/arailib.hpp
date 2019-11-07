#ifndef ARAILIB_ARAILIB_HPP
#define ARAILIB_ARAILIB_HPP

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

    typedef std::vector<Point> Series;

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

    float angular_distance(const Point& p1, const Point& p2) {
        return acos(cosine_similarity(p1, p2)) / static_cast<float>(M_PI);
    }

    template <class T = float>
    std::vector<T> split(std::string &input, char delimiter = ',') {
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
}

#endif //ARAILIB_ARAILIB_HPP