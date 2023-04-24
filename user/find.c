#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fs.h"

void find(char* path, char* target){
    char buf[512], *p;
    int fd;
    struct dirent de;
    struct stat st;
    //抄ls.c.的
    if((fd = open(path, 0)) < 0){
        fprintf(2, "find: cannot open %s\n", path);
        return;
    }

    if(fstat(fd, &st) < 0){
        fprintf(2, "find: cannot stat %s\n", path);
        close(fd);
        return;
    }
    //继续抄
    switch(st.type){
    case T_FILE:
    //对传入参数进行判断，path就是argv[1]，+-防止溢出，实际指针向前移动到在对齐target长度的path位置
        if(strcmp(path+strlen(path)-strlen(target),target) == 0){
            printf("%s\n",path);
        }
        break;
    //继续抄
    case T_DIR:
        if(strlen(path) + 1 + DIRSIZ + 1 > sizeof buf){
            printf("ls: path too long\n");
            break;
        }
        strcpy(buf, path);
        p = buf+strlen(buf);
        *p++ = '/';
        while(read(fd, &de, sizeof(de)) == sizeof(de)){
            if(de.inum == 0)
                continue;
            memmove(p, de.name, DIRSIZ);
            p[DIRSIZ] = 0;
            if(stat(buf, &st) < 0){
                printf("find: cannot stat %s\n", buf);
                continue;
            }
            //对接下来的字符进行判断，不进入.和..   之后继续find里面的文件
            if(strcmp(buf+strlen(buf)-2, "/.") != 0 && strcmp(buf+strlen(buf)-3, "/..") != 0l){
                find(buf, target);
            }
        }
        break;
    }
    close(fd);
}
int main(int argc, char* argv[]){//传入参数变成字符
    if(argc < 3){//3个操作数
        exit(0);
    }
    char target[512];
    target[0] = '/';//设定第一个为“/”
    strcpy(target+1, argv[2]);//把1号指针（覆盖）指向传入操作参数
    find(argv[1], target);//传入操作内容和target
    exit(0);
}