//
// Created by unsky on 09/08/18.
//

#ifndef SITA_STUFF_STUFF_H
#define SITA_STUFF_STUFF_H
#include "shuffle/fisher_yates_shuffler.h"
namespace  sita {

class Stuff {
public:
    Stuff() {}
    ~Stuff() {}

    virtual void exec_infer() {}

    virtual void exec_back() {}

private:
    DISABLE_COPY_AND_ASSIGN(Stuff);
};
}
#endif //SITA_STUFF_H
