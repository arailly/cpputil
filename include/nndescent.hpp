//
// Created by Yusuke Arai on 2019-09-09.
//

#ifndef ARAILIB_NNDESCENT_HPP
#define ARAILIB_NNDESCENT_HPP

#include <random>
#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include "arailib.hpp"

namespace arailib::nndescent {

    struct Neighbor {
        Point query;
        Point point;
        float distance;

        Neighbor(Point& q, Point& p, float d) : query(q), point(p), distance(d) {}

        Neighbor(Point& q, Point& p) : query(q), point(p), distance(l2_norm(query, point)) {}

    };

    bool operator<(const Neighbor& n1, const Neighbor& n2) { return n1.distance < n2.distance; }
    bool operator>(const Neighbor& n1, const Neighbor& n2) { return n1.distance > n2.distance; }
    bool operator<=(const Neighbor& n1, const Neighbor& n2) { return n1.distance <= n2.distance; }
    bool operator>=(const Neighbor& n1, const Neighbor& n2) { return n1.distance >= n2.distance; }

    typedef std::vector<Neighbor> Neighbors;

    class KNearestNeighborsHeap {
    public:
        size_t id;
        size_t k = 0;
        Point query;
        std::priority_queue<Neighbor> neighbor_heap;

        KNearestNeighborsHeap(size_t k, Point q) : k(k), query(q) {}

        bool initialized() { return k != 0; }

        void update(Point& point) {
            Neighbor n(query, point);
            neighbor_heap.push(n);
        }

        Neighbor furthest() const { return neighbor_heap.top(); }

        void pop() { neighbor_heap.pop(); }

        bool empty() const { return neighbor_heap.empty(); }

        Neighbors vectorize() const {
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

    typedef std::vector<KNearestNeighborsHeap> KNearestNeighborsHeapList;

    KNearestNeighborsHeap sample(Series& series, Point& query, size_t n_sample, u_int random_state=42) {
        std::mt19937 engine(random_state);
        std::uniform_int_distribution<> dist(0, static_cast<int>(series.size() - 1));

        std::map<size_t, bool> random_id_map;
        KNearestNeighborsHeap knn(n_sample, query);
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

    typedef std::vector<Series> ReverseKNearestNeighborsList;

    ReverseKNearestNeighborsList reverse(KNearestNeighborsHeapList& knn_list) {
        auto k = knn_list[0].k;
        std::vector<Series> reverse_knn_list(knn_list.size());
        for (auto knn : knn_list) {
            while (!knn.empty()) {
                auto n = knn.furthest();
                reverse_knn_list[n.point.id].push_back(knn.query);
                knn.pop();
            }
        }
        return reverse_knn_list;
    }

    KNearestNeighborsHeapList create_knn_graph(Series& series, size_t k, int random_state=42) {
         KNearestNeighborsHeapList knn_list;
         for (auto& query : series) {
             KNearestNeighborsHeap knn = sample(series, query, k);
             knn_list.push_back(knn);
         }

         while (true) {
              auto reverse_knn_list = reverse(knn_list);
             // auto local_join_list = local_join(knn_list, reverse_knn_list);
             // bool updated = false;
         }

         return knn_list;
    }

}

#endif //ARAILIB_NNDESCENT_HPP
