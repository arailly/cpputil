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
#include "nlohmann/json.hpp"

using namespace std;
using namespace nlohmann;

namespace arailib {

template <class UnaryOperation, class Iterable>
Iterable fmap(UnaryOperation op, const Iterable &v) {
    Iterable result;
    std::transform(v.begin(), v.end(), std::back_inserter(result), op);
    return result;
}

template <class Predicate, class Iterable>
Iterable filter(Predicate pred, const Iterable &v) {
    Iterable result;
    std::copy_if(v.begin(), v.end(), std::back_inserter(result), pred);
    return result;
}

template <class Iterable>
float l2_norm(const Iterable& v1, const Iterable& v2) {
    float result = 0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += std::pow(v1[i] - v2[i], 2);
    }
    result = std::sqrt(result);
    return result;
}

struct Point {
    size_t id;
    std::vector<double> x;

    Point(size_t i, std::vector<double> v) {
        id = i;
        std::copy(v.begin(), v.end(), std::back_inserter(x));
    }

    Point(std::vector<double> v) {
        id = 0;
        std::copy(v.begin(), v.end(), std::back_inserter(x));
    }

    double& operator[] (size_t i) { return x[i]; }
    const double& operator[] (size_t i) const { return x[i]; }

    bool operator==(const Point& o) const {
        if (id == o.id) return true;
        return false;
    }

    bool operator!=(const Point& o) const {
        if (id != o.id) return true;
        return false;
    }

    size_t size() const { return x.size(); }
    auto begin() const { return x.begin(); }
    auto end() const { return x.end(); }

    void show() const {
        std::cout << id << ": ";
        for (const auto& xi : x) {
            std::cout << xi << ' ';
        } std::cout << std::endl;
    }
};

typedef std::vector<Point> Series;

auto split(std::string& input, char delimiter=',') {
    std::istringstream stream(input);
    std::string field;
    std::vector<double> result;

    while (std::getline(stream, field, delimiter)) {
        result.push_back(std::stod(field));
    }

    return result;
}

Series read_csv(const std::string& path, const size_t& nrows=-1,
                 const bool& skip_header=false) {
    std::ifstream ifs(path);
    if (!ifs) throw "Can't open file!";
    std::string line;

    Series series;
    for (size_t i = 0; (i < nrows) && std::getline(ifs, line); ++i) {
        // if first line is the header then skip
        if (skip_header && (i == 0)) continue;
        std::vector<double> v = split(line);
        series.push_back(Point(i, v));
    }
    return series;
}

template <typename T>
void write_csv(const std::vector<T>& v, const std::string& path) {
    std::ofstream ofs(path);
    for (const auto& o : v) {
        std::string line;
        for (const auto& e : o) {
            line += std::to_string(e) + ',';
        }
        line.pop_back();
        line += '\n';
        ofs << line;
    }
}

json read_config(const string config_path = "./config.json") {
    json config;
    ifstream ifs(config_path);
    if (ifs.fail()) throw "Error: config.json not found.";
    ifs >> config;
    return config;
}

}

#endif //ARAILIB_ARAILIB_HPP