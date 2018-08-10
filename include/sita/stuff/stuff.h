//
// Created by unsky on 09/08/18.
//

#ifndef SITA_STUFF_STUFF_H
#define SITA_STUFF_STUFF_H
namespace  sita {

class Stuff {
public:

    Stuff() {}
    ~Stuff() {}

    virtual void inferance() {}

    virtual void backward() {}

private:
    DISABLE_COPY_AND_ASSIGN(Stuff);
};
}
#endif //SITA_STUFF_H
