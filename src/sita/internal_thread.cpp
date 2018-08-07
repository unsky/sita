//
// Created by unsky on 07/08/18.
//

#include "sita/internal_thread.h"

namespace sita {
    InternalThread::~InternalThread() {
        stop_internal_thread();
    }

    bool InternalThread::is_started() const {
        return thread_ && thread_->joinable();
    }

    bool InternalThread::must_stop() {
        return thread_ && thread_->interruption_requested();
    }

    void InternalThread::start_internal_thread() {
        CHECK(!is_started()) << "Threads should persist and not be restarted.";

        try {
            thread_.reset(new boost::thread(&InternalThread::entry, this));
        } catch (std::exception& e) {
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }

    void InternalThread::entry() {
        internal_thread_entry();
    }

    void InternalThread::stop_internal_thread() {
        if (is_started()) {
            thread_->interrupt();
            try {
                thread_->join();
            } catch (boost::thread_interrupted&) {
            } catch (std::exception& e) {
                LOG(FATAL) << "Thread exception: " << e.what();
            }
        }
    }

}  // namespace sita
