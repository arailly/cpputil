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

        Neighbor(const Point& q, const Point& p, float d) : query(q), point(p), distance(d) {}

        Neighbor(const Point& q, const Point& p) : query(q), point(p), distance(l2_norm(query, point)) {}
    };

    bool operator<(const Neighbor& n1, const Neighbor& n2) { return n1.distance < n2.distance; }
    bool operator>(const Neighbor& n1, const Neighbor& n2) { return n1.distance > n2.distance; }
    bool operator<=(const Neighbor& n1, const Neighbor& n2) { return n1.distance <= n2.distance; }
    bool operator>=(const Neighbor& n1, const Neighbor& n2) { return n1.distance >= n2.distance; }
    bool operator==(const Neighbor& n1, const Neighbor& n2) { return n1.point == n2.point; }
    bool operator!=(const Neighbor& n1, const Neighbor& n2) { return n1.point != n2.point; }

    typedef std::vector<Neighbor> Neighbors;

    class KNNHeap {
    public:
        size_t k = 0;
        const Point query;
        std::priority_queue<Neighbor> neighbor_heap;
        std::map<size_t, bool> added;
        bool flag = true;

        KNNHeap(size_t k, const Point q) : k(k), query(q) {
            added[q.id] = true; // self
        }

        bool initialized() { return k != 0; }

        bool update(const Point& point) {
            Neighbor n(query, point);

            if (neighbor_heap.size() == 0) {
                added[point.id] = true;
                neighbor_heap.push(n);
                return true;
            }

            if (added[point.id]) return false;
            added[point.id] = true;

            auto previous_furthest = furthest();

            neighbor_heap.push(n);
            if (size() > k) neighbor_heap.pop();

            return (previous_furthest != furthest());
        }

        bool update(const Series& series) {
            bool updated = false;
            for (const auto& point : series) {
                updated = update(point) || updated;
            }
            return updated;
        }

        Neighbor furthest() const { return neighbor_heap.top(); }

        void pop() { neighbor_heap.pop(); }

        bool empty() const { return neighbor_heap.empty(); }

        size_t size() const { return neighbor_heap.size(); }

        Series get_knn_series(bool ascending=false) const {
            Series series;
            auto neighbor_heap_copy = neighbor_heap;
            while (!neighbor_heap_copy.empty()) {
                series.push_back(neighbor_heap_copy.top().point);
                neighbor_heap_copy.pop();
            }

            if (ascending) std::reverse(series.begin(), series.end());
            return series;
        }
    };

    typedef std::vector<KNNHeap> KNNHeapList;

    Series sample(const Series& series, const Point& query, size_t n_sample, u_int random_state=42) {
        std::mt19937 engine(random_state);
        std::uniform_int_distribution<> dist(0, static_cast<int>(series.size() - 1));

        std::map<size_t, bool> random_id_map;
        Series result;
        for (size_t i = 0; i < n_sample; i++) {
            auto random_id = static_cast<size_t>(dist(engine));

            if (random_id_map[random_id] || query.id == random_id) i--;
            else {
                random_id_map[random_id] = true;
                result.push_back(series[random_id]);
            }
        }
        return result;
    }

    typedef std::vector<Series> SeriesList;

    SeriesList reverse(const KNNHeapList& knn_list) {
        auto k = knn_list[0].k;
        SeriesList reverse_knn_list(knn_list.size());
        for (auto knn : knn_list) {
            while (!knn.empty()) {
                auto n = knn.furthest();
                reverse_knn_list[n.point.id].push_back(knn.query);
                knn.pop();
            }
        }
        return reverse_knn_list;
    }

    SeriesList local_join(const KNNHeapList& knn_list, const SeriesList& rknn_list) {
        SeriesList local_join_list;

        for (size_t i = 0; i < knn_list.size(); i++) {
            Series series;
            std::map<size_t, bool> added;
            added[i] = true; // self

            const auto& knn = knn_list[i];
            const auto&& knn_vector = knn.get_knn_series();

            for (const auto& e : knn_vector) {
                if (added[e.id]) continue;
                added[e.id] = true;
                series.push_back(e);
            }

            const auto& rknn = rknn_list[i];
            for (const auto& e : rknn) {
                if (added[e.id]) continue;
                added[e.id] = true;
                series.push_back(e);
            }

            local_join_list.push_back(series);
        }
        return local_join_list;
    }

    KNNHeapList create_knn_graph_naive(Series& series, size_t k, int random_state=42) {
         KNNHeapList knn_list;
         for (auto& query : series) {
             KNNHeap knn(k, query);
             knn.update(sample(series, query, k));
             knn_list.push_back(knn);
         }

         while (true) {
             auto&& reverse_knn_list = reverse(knn_list);
             auto&& local_join_list = local_join(knn_list, reverse_knn_list);
             int n_updated = 0;
             for (const auto& point : series) {
                 std::map<size_t, bool> added;
                 added[point.id] = true; // self
                 for (auto& u1: local_join_list[point.id]) {
                     for (auto& u2: local_join_list[u1.id]) {
                         if (added[u2.id]) continue;
                         added[u2.id] = true;
                         n_updated += knn_list[point.id].update(u2);
                     }
                 }
             }
             if (n_updated == 0) return knn_list;
         }
    }

    KNNHeapList create_knn_graph(const Series& series, size_t k,
            float rho, float delta, int random_state=42) {
        KNNHeapList knn_list;
        for (auto& query : series) {
            KNNHeap knn(k, query);
            knn.update(sample(series, query, k));
            knn_list.push_back(knn);
        }
    }
}

#endif //ARAILIB_NNDESCENT_HPP
