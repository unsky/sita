//
// Created by cs on 10/08/18.
//

#include "sita/stuff/shuffle/fisher_yates_shuffler.h"
namespace  sita{
int64_t FisherYatesShuffler::cluster_seedgen() {
    int64_t s, seed, pid;
    FILE* f = fopen("/dev/urandom", "rb");
    if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
        fclose(f);
        return seed;
    }

    LOG(INFO) << "System entropy source not available, "
                 "using fallback algorithm to generate seed instead.";
    if (f)
        fclose(f);

    pid = getpid();
    s = time(NULL);
    seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
    return seed;
}

}