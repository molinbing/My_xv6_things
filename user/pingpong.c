#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

int main(int argc, int* argv[]){//标准输入：argc argv[][]
    int poutf[2], pinc[2];//创造俩指针，0口写，1口读
    pipe(poutf);//创建管道
    pipe(pinc);

    if(fork() != 0){//创建子线程，对子线程判断
        write(poutf[1],"6",1);
        char waiti;
        read(pinc[0],&waiti,1);
        printf("%d: received pong\n",getpid());
        wait(0);
    }else{
        char geti;
        read(poutf[0],&geti,1);
        printf("%d: received pong\n",getpid());
        sleep(5);
        write(pinc[1],&geti,1);

    }
    exit(0);
}