//
// Created by cs on 06/08/18.
//

#include "sita/stuff/io_protobuff.h"

namespace sita{

    bool ReadProtoFromTextFile(const char* filename, Message* proto) {
        int fd = open(filename, O_RDONLY);
        CHECK_NE(fd, -1) << "File not found: " << filename;
        FileInputStream* input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }
}