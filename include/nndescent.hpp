//
// Created by Yusuke Arai on 2019-09-09.
//

#ifndef ARAILIB_NNDESCENT_HPP
#define ARAILIB_NNDESCENT_HPP

#include <random>
#include <vector>
#include <queue>
#include "arailib.hpp"

namespace arailib::nndescent {

    struct Neighbor {
        size_t id;
        float distance;

        Neighbor(size_t i, float d) : id(i), distance(d) {}

        Neighbor(const Object& query, const Object& point) {
            id = point.id;
            distance = l2_norm(query, point);
        }

    };

    bool operator<(const Neighbor& n1, const Neighbor& n2) { return n1.distance < n2.distance; }
    bool operator>(const Neighbor& n1, const Neighbor& n2) { return n1.distance > n2.distance; }
    bool operator<=(const Neighbor& n1, const Neighbor& n2) { return n1.distance <= n2.distance; }
    bool operator>=(const Neighbor& n1, const Neighbor& n2) { return n1.distance >= n2.distance; }

    typedef std::vector<Neighbor> Neighbors;

    class KNearestNeighbors {
    public:
        size_t k;
        Object query;
        std::priority_queue<Neighbor> neighbor_heap;

        KNearestNeighbors(size_t k, const Object& o) : k(k), query(o) {}

        void update(const Object& point) {
            Neighbor n(query, point);
            neighbor_heap.push(n);
        }

        Neighbor furthest() const {
            return neighbor_heap.top();
        }

        void pop() {
            neighbor_heap.pop();
        }
    };

    typedef std::vector<KNearestNeighbors> KNearestNeighborsList;

    KNearestNeighbors sample(const Series& series, const Object& query, size_t n_sample, int random_state = 42) {
        std::mt19937 engine(random_state);
        std::uniform_int_distribution<> dist(0, series.size() - 1);

         KNearestNeighbors knn(n_sample, query);
         for (size_t i = 0; i < n_sample; i++) {
           auto random_number = static_cast<size_t>(dist(engine));
           knn.update(series[random_number]);
         }
         return knn;
    }

    KNearestNeighborsList nn_descent(const Series& series, size_t k, int random_state=42) {
        // KNearestNeighborsList knn_list;
        // for (const auto& object : series) {
        //   KNearestNeighbors knn = sample(series, query, k);
        //   knn_list.push_back(knn);
        // }
    }

}

#endif //ARAILIB_NNDESCENT_HPP
