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
double l2_norm(const Iterable& v1, const Iterable& v2) {
    double result = 0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += std::pow(v1[i] - v2[i], 2);
    }
    result = std::sqrt(result);
    return result;
}

struct Object {
    size_t index;
    std::vector<double> x;

    Object(size_t i, std::vector<double> v) {
        index = i;
        std::copy(v.begin(), v.end(), std::back_inserter(x));
    }

    Object(std::vector<double> v) {
        index = 0;
        std::copy(v.begin(), v.end(), std::back_inserter(x));
    }

    double& operator[] (size_t i) { return x[i]; }
    const double& operator[] (size_t i) const { return x[i]; }

    bool operator==(const Object& o) const {
        if (index == o.index) return true;
        return false;
    }

    bool operator!=(const Object& o) const {
        if (index != o.index) return true;
        return false;
    }

    size_t size() const { return x.size(); }
    auto begin() const { return x.begin(); }
    auto end() const { return x.end(); }

    void show() const {
        std::cout << index << ": ";
        for (const auto& xi : x) {
            std::cout << xi << ' ';
        } std::cout << std::endl;
    }
};

typedef std::vector<Object> Series;

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
    std::string line;

    Series series;
    for (size_t i = 0; (i < nrows) && std::getline(ifs, line); ++i) {
        // if first line is the header then skip
        if (skip_header && (i == 0)) continue;
        std::vector<double> v = split(line);
        series.push_back(Object(i, v));
    }
    return series;
}

}

#endif //ARAILIB_ARAILIB_HPP