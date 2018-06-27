//
// Created by unsky on 27/06/18.
//

#ifndef STBA_STUFF_COMMON_H
#define STBA_STUFF_COMMON_H

#include <vector>
#include <glog/logging.h>
namespace stba {
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"


}//namespace
#endif //STBA_STUFF_COMMON_H
