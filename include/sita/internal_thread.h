//
// Created by unsky on 07/08/18.
//

#ifndef SITA_THREAD_CONTROL_H
#define SITA_THREAD_CONTROL_H
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include <exception>

namespace boost { class thread; }
namespace sita {
class InternalThread {
public:
    InternalThread() : _thread() {}
    virtual ~InternalThread();

    void start_internal_thread();

    void stop_internal_thread();

    bool is_started() const;

protected:
    virtual  void internal_thread_entry(){};
    bool must_stop();

private:
    void entry();
    boost::shared_ptr<boost::thread> _thread;
};

}  // namespace caffe

#endif //SITA_THREAD_CONTROL_H
