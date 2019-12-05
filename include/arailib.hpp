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
#include <chrono>
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

    auto get_now() { return chrono::system_clock::now(); }

    auto get_duration(chrono::system_clock::time_point start,
                      chrono::system_clock::time_point end) {
        return chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    bool is_csv(const string& path) {
        return (path.rfind(".csv", path.size()) < path.size());
    }
}

#endif //ARAILIB_ARAILIB_HPP