//
// Created by unsky on 06/08/18.
//

#include "sita/io_protobuff.h"

namespace sita{

    bool read_proto_from_txt(const char* filename, Message* proto) {
        int fd = open(filename, O_RDONLY);
        CHECK_NE(fd, -1) << "File not found: " << filename;
        FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }
}