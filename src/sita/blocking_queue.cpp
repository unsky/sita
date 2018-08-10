//
// Created by unsky on 07/08/18.
//
#include "sita/blocking_queue.h"
#include "sita/dataprovider/mnist_dataprovider.h"

namespace sita {

template<typename T>
class BlockingQueue<T>::sync {
public:
    mutable boost::mutex _mutex;
    boost::condition_variable _condition;

};

template<typename T>
BlockingQueue<T>::BlockingQueue()
        : _sync(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
    boost::mutex::scoped_lock lock(_sync->_mutex);
    _queue.push(t);
    lock.unlock();
    _sync->_condition.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
    boost::mutex::scoped_lock lock(_sync->_mutex);

    if (_queue.empty()) {
        return false;
    }

    *t = _queue.front();
    _queue.pop();
    return true;
}

template<typename T>
T BlockingQueue<T>::pop(const std::string& log_on_wait) {
    boost::mutex::scoped_lock lock(_sync->_mutex);

    while (_queue.empty()) {
        if (!log_on_wait.empty()) {
            LOG_EVERY_N(INFO, 1000)<< log_on_wait;
        }
        _sync->_condition.wait(lock);
    }

    T t = _queue.front();
    _queue.pop();
    return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
    boost::mutex::scoped_lock lock(_sync->_mutex);

    if (_queue.empty()) {
        return false;
    }

    *t = _queue.front();
    return true;
}

template<typename T>
T BlockingQueue<T>::peek() {
    boost::mutex::scoped_lock lock(_sync->_mutex);

    while (_queue.empty()) {
        _sync->_condition.wait(lock);
    }

    return _queue.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
    boost::mutex::scoped_lock lock(_sync->_mutex);
    return _queue.size();
}

template class BlockingQueue<MnistBatch<float >*>;
template class BlockingQueue<MnistBatch<double >*>;
template class BlockingQueue<MnistBatch<int >*>;


}  // namespace sita
