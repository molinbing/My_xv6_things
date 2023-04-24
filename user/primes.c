#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

void ziin(int input[]){
    int p;
    read(input[0],&p,sizeof(p));//从父（相对）线程传入的管道读
    if(p == -1){//传入-1即为已经完成
        exit(0);
    }
    printf("prime %d\n",p);//每个子线程对第一个传入的必然为质数的值打印

    int output[2];//创建管道给下一个用
    pipe(output);

    if(fork() == 0){
        close(output[1]);//关写
        close(input[0]);//关读
        ziin(output);//传入上面刚刚创建的管道
    }else{
        close(output[0]);
        int buf;
        while(read(input[0],&buf,sizeof(buf)) && buf != -1){
            if(buf % p != 0){
                write(output[1],&buf,sizeof(buf));
            }
        }
        buf = -1;
        write(output[1],&buf,sizeof(buf));
        wait(0);
        exit(0);
    }
}

int main(int argc, int* argv[]){
    int input[2];
    pipe(input);

    if(fork() == 0){
        close(input[1]);
        ziin(input);//用子线程打印
        exit(0);
    }else{
        close(input[0]);
        for(int i =2; i<=35; i++){
            write(input[1],&i,sizeof(i));
        }
        int i = -1;
        write(input[1],&i,sizeof(i));
    }
    wait(0);
    exit(0);
}

