//
// Created by cs on 07/08/18.
//

#ifndef SITA_THREAD_CONTROL_H
#define SITA_THREAD_CONTROL_H
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <exception>

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class thread; }

namespace sita {

/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virtual function InternalThreadEntry.
 */
    class InternalThread {
    public:
        InternalThread() : thread_() {}
        virtual ~InternalThread();

        /**
         * Caffe's thread local state will be initialized using the current
         * thread values, e.g. device id, solver index etc. The random seed
         * is initialized using caffe_rng_rand.
         */
        void start_internal_thread();
       // void StartInternalThread();

        /** Will not return until the internal thread has exited. */
        void stop_internal_thread();
       // void StopInternalThread();

        bool is_started() const;

    protected:
        /* Implement this method in your subclass
            with the code you want your thread to run. */
       // virtual void InternalThreadEntry() {}
        virtual  void internal_thread_entry(){}
        /* Should be tested when running loops to exit when requested. */
        bool must_stop();

    private:
        void entry();
        boost::shared_ptr<boost::thread> thread_;
    };

}  // namespace caffe

#endif //SITA_THREAD_CONTROL_H
