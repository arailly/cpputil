#ifndef ARAILIB_ARAILIB_HPP
#define ARAILIB_ARAILIB_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>

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
        for (int i = 0; i < v1.size(); i++) {
            result += std::pow(v1[i] - v2[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

}

#endif //ARAILIB_ARAILIB_HPP