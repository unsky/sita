//
// Created by unsky on 07/08/18.
//

#ifndef SITA_BLOCK_QUEUE_H
#define SITA_BLOCK_QUEUE_H

#include <queue>
#include <string>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "sita/tensor.h"

namespace sita {

template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue();

    void push(const T& t);

    bool try_pop(T* t);

    // This logs a message if the threads needs to be blocked
    // useful for detecting e.g. when data feeding is too slow
    T pop(const std::string& log_on_wait = "");

    bool try_peek(T* t);

    // Return element without removing it
    T peek();

    size_t size() const;

protected:
    class sync;

    std::queue<T> queue_;
    boost::shared_ptr<sync> sync_;

    DISABLE_COPY_AND_ASSIGN(BlockingQueue);
};

}  // namespace caffe
#endif //SITA_BLOCK_QUEUE_H
