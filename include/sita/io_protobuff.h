//
// Created by unsky on 06/08/18.
//

#ifndef SITA_IO_PROTOBUFF_H
#define SITA_IO_PROTOBUFF_H

#include "google/protobuf/message.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <glog/logging.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <boost/filesystem.hpp>
#include <iomanip>

namespace  sita{
    using ::google::protobuf::Message;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    using google::protobuf::io::CodedInputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    using google::protobuf::io::CodedOutputStream;
    using google::protobuf::Message;
    bool ReadProtoFromTextFile(const char* filename, Message* proto);
}

#endif //SITA_IO_PROTOBUFF_H
