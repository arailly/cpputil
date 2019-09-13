//
// Created by Yusuke Arai on 2019-09-09.
//

#ifndef ARAILIB_NNDESCENT_HPP
#define ARAILIB_NNDESCENT_HPP

#include <random>
#include <vector>
#include <map>
#include <queue>
#include "arailib.hpp"

namespace arailib::nndescent {

    struct Neighbor {
        size_t id;
        float distance;

        Neighbor(size_t i, float d) : id(i), distance(d) {}

        Neighbor(const Point& query, const Point& point) {
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
        Point query;
        std::priority_queue<Neighbor> neighbor_heap;

        KNearestNeighbors(size_t k, const Point& o) : k(k), query(o) {}

        void update(const Point& point) {
            Neighbor n(query, point);
            neighbor_heap.push(n);
        }

        Neighbor furthest() const {
            return neighbor_heap.top();
        }

        void pop() {
            neighbor_heap.pop();
        }

        std::vector<Neighbor> vectorize() const {
            auto neighbor_heap_copy = neighbor_heap;
            std::vector<Neighbor> result;
            while (!neighbor_heap_copy.empty()) {
                result.push_back(neighbor_heap_copy.top());
                neighbor_heap_copy.pop();
            }
            std::reverse(result.begin(), result.end());
            return result;
        }
    };

    typedef std::vector<KNearestNeighbors> KNearestNeighborsList;

    KNearestNeighbors sample(const Series& series, const Point& query, size_t n_sample, int random_state=42) {
        std::mt19937 engine(random_state);
        std::uniform_int_distribution<> dist(0, series.size() - 1);

        std::map<size_t, bool> random_id_map;
        KNearestNeighbors knn(n_sample, query);
        for (size_t i = 0; i < n_sample; i++) {
            auto random_id = static_cast<size_t>(dist(engine));

            if (random_id_map[random_id] || query.id == random_id) i--;
            else {
                random_id_map[random_id] = true;
                knn.update(series[random_id]);
            }
        }
        return knn;
    }

    KNearestNeighborsList reverse(KNearestNeighborsList knn_list) {
        KNearestNeighborsList reverse_knn_list;
        for (const auto& knn : knn_list) {

        }
        return reverse_knn_list;
    }

    KNearestNeighborsList create_knn_graph(const Series& series, size_t k, int random_state=42) {
         KNearestNeighborsList knn_list;
         for (const auto& query : series) {
             KNearestNeighbors knn = sample(series, query, k);
             knn_list.push_back(knn);
         }

         while (true) {
             // auto reverse_knn_list = reverse(knn_list);
             // auto local_join_list = local_join(knn_list, reverse_knn_list);
             // bool updated = false;
         }

         return knn_list;
    }

}

#endif //ARAILIB_NNDESCENT_HPP
