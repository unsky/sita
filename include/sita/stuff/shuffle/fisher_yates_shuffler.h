//
// Created by unsky on 09/08/18.
//

#ifndef SITA_STUFF_RNG_RNG_H
#define SITA_STUFF_RNG_RNG_H

#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <glog/logging.h>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "sita/stuff/stuff.h"
namespace sita{

class FisherYatesShuffler: public Stuff{

public:
int64_t cluster_seedgen();

//inline void exec_infer(){}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
             RandomGenerator* gen){
    typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
            difference_type;
    typedef typename boost::uniform_int<difference_type> dist_type;

    difference_type length = std::distance(begin, end);
    if (length <= 0) return;

    for (difference_type i = length - 1; i > 0; --i) {
        dist_type dist(0, i);
        std::iter_swap(begin + i, begin + dist(*gen));
    }
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end){
    int64_t seed;
    seed = cluster_seedgen();
    boost::mt19937 rng(seed);
    shuffle(begin, end, &rng);
}
};







}
#endif //SITA_RNG_H
