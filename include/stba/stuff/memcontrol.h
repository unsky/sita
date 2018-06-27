//
// Created by unsky on 27/06/18.
//

#ifndef STBA_STUFF_MEMCONTROL_H
#define STBA_STUFF_MEMCONTROL_H

#include <cstdlib>
#include "stba/stuff/common.h"

namespace stba{
    class MemContol{
        public:
            MemControl();
            ~MemControl();

            void * mutable_gpu_data();
            void * mutable_gpu_data();
            const void * cpu_data();
            const void * gpu_data();
            inline size_t size(){
                return _size;
            }
            enum  HeadAt{UNINIT, CPU, GPU, SYNCED};
            inline HeadAt head_at()
            {
                return _head_at;
            }

        private:
            void * _ptr_cpu;
            void * _ptr_gpu;
            void push_data_to_cpu();
            void push_data_to_gpu();
            size_t _size;
            HeadAt _head_at;
            DISABLE_COPY_AND_ASSIGN(MemControl);
    };

}//namespace

#endif //STBA_MEMCONTROL_H
