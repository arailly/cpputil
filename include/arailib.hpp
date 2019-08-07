#ifndef ARAILIB_ARAILIB_HPP
#define ARAILIB_ARAILIB_HPP

namespace arailib {

    template<class UnaryOperation, class Iterable>
    Iterable fmap(UnaryOperation op, Iterable &v) {
        Iterable result;
        std::transform(v.begin(), v.end(), std::back_inserter(result), op);
        return result;
    }

    template<class Predicate, class Iterable>
    Iterable filter(Predicate pred, Iterable &v) {
        Iterable result;
        std::copy_if(v.begin(), v.end(), std::back_inserter(result), pred);
        return result;
    }

}

#endif //ARAILIB_ARAILIB_HPP