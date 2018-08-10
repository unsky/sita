//
// Created by unsky on 07/08/18.
//

#include "sita/internal_thread.h"

namespace sita {

InternalThread::~InternalThread() {
    stop_internal_thread();
}

bool InternalThread::is_started() const {
    return _thread && _thread->joinable();
}

bool InternalThread::must_stop() {
    return _thread && _thread->interruption_requested();
}

void InternalThread::start_internal_thread() {
    CHECK(!is_started()) << "Threads should persist and not be restarted.";

    try {
        _thread.reset(new boost::thread(&InternalThread::entry, this));
    } catch (std::exception& e) {
        LOG(FATAL) << "Thread exception: " << e.what();
    }
}

void InternalThread::entry() {
    internal_thread_entry();
}

void InternalThread::stop_internal_thread() {
    //fist thread try to join the main and interupted itself
    if (is_started()) {
        _thread->interrupt();
        try {
            _thread->join();
        } catch (boost::thread_interrupted&) {
        } catch (std::exception& e) {
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }
    //second third,..thread interrupt
    if(_thread){
        _thread->interrupt();
    }
}

}  // namespace sita
