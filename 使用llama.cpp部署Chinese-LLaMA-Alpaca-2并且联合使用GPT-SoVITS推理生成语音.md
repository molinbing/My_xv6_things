*本教程使用wsl2-Ubuntu2204进行部署，其他环境请参考[llamacpp_zh · ymcui/Chinese-LLaMA-Alpaca-2 Wiki (github.com)](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_zh)和[GPT-SoVITS/docs/cn/README.md at main · RVC-Boss/GPT-SoVITS (github.com)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md)*
# Part 1：使用llama.cpp部署Chinese-LLaMA-Alpaca-2
*以下引用**[原文档](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_zh)**教程*
>以[llama.cpp工具](https://github.com/ggerganov/llama.cpp)为例，介绍模型量化并在本地部署的详细步骤。Windows则可能需要cmake等编译工具的安装。**本地快速部署体验推荐使用经过指令精调的Alpaca-2模型，有条件的推荐使用6-bit或者8-bit模型，效果更佳。** 运行前请确保：
>
>1. 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
2. 建议使用Python 3.10以上编译和运行该工具
>
>### Step 1: 克隆和编译llama.cpp[](https://github.com/ymcui/Chinese-Mixtral/wiki/llamacpp_zh#step-1-%E5%85%8B%E9%9A%86%E5%92%8C%E7%BC%96%E8%AF%91llamacpp)
>
>1. （可选）如果已下载旧版仓库，建议`git pull`拉取最新代码，**并执行`make clean`进行清理**
2. 拉取最新版llama.cpp仓库代码
>
>```shell
$ git clone https://github.com/ggerganov/llama.cpp
>```
>
>2. 对llama.cpp项目进行编译，生成`./main`（用于推理）和`./quantize`（用于量化）二进制文件。
>
>```shell
$ make
>```
>
>**Linux用户**如需启用GPU推理，则推荐与[BLAS（或cuBLAS如果有GPU）一起编译](https://github.com/ggerganov/llama.cpp#blas-build)，可以提高prompt处理速度。以下是和cuBLAS一起编译的命令，适用于NVIDIA相关GPU。参考：[llama.cpp#blas-build](https://github.com/ggerganov/llama.cpp#blas-build)
>
>```shell
$ make LLAMA_CUBLAS=1
>```
>
>### Step 2: 生成量化版本模型[](https://github.com/ymcui/Chinese-Mixtral/wiki/llamacpp_zh#step-2-%E7%94%9F%E6%88%90%E9%87%8F%E5%8C%96%E7%89%88%E6%9C%AC%E6%A8%A1%E5%9E%8B)
>
>（💡 也可直接下载已量化好的gguf模型：[gguf模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/tree/main?tab=readme-ov-file#%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)）
>
>目前llama.cpp已支持`.pth`文件以及huggingface格式`.bin`的转换。将完整模型权重转换为GGML的FP16格式，生成文件路径为`zh-models/7B/ggml-model-f16.gguf`。进一步对FP16模型进行4-bit量化，生成量化模型文件路径为`zh-models/7B/ggml-model-q4_0.gguf`。
>
>```shell
$ python convert.py zh-models/7B/
$ ./quantize ./zh-models/7B/ggml-model-f16.gguf ./zh-models/7B/ggml-model-q4_0.gguf q4_0
>```

这里以下载了[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2?tab=readme-ov-file#%E5%AE%8C%E6%95%B4%E6%A8%A1%E5%9E%8B%E4%B8%8B%E8%BD%BD)非GGUF模型）：
1.请将你下载好的模型文件放在你的`llama.cpp/zh-models/7B/`内
2.在llama.cpp文件夹内运行上述指令`$ python convert.py zh-models/7B/`时你可能遇到类似如下的报错：
```python
$ python convert.py zh-models/7B/
Traceback (most recent call last):
  File "~/llama.cpp/convert.py", line 27, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```
请自行安装numpy库或其他缺失的库：
```python
$ pip install numpy
```
然后继续执行并且量化模型

*另：如果你下载了**GGUF版**模型，请直接放入量化模型文件路径（zh-models/7B/或其他）下即可。*
### Step 3:加载并启动模型
*你可以将本项目的`scripts/llama-cpp/chat.sh`拷贝至llama.cpp的根目录。*
或请在llama.cpp文件夹内执行：
```bash
$ vim chat.sh
```
点击`i`并写入如下内容后点击Esc并输入`:wq`保存退出：
```
#!/bin/bash

# temporary script to chat with Chinese Alpaca-2 model
# usage: ./chat.sh alpaca2-ggml-model-path your-first-instruction

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
# SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
MODEL_PATH=$1
FIRST_INSTRUCTION=$2

./main -m "$MODEL_PATH" \
--color -i -c 4096 -t 8 --temp 0.5 --top_k 40 --top_p 0.9 --repeat_penalty 1.1 \
--in-prefix-bos --in-prefix ' [INST] ' --in-suffix ' [/INST]' -p \
"[INST] <<SYS>>
$SYSTEM_PROMPT
<</SYS>>

$FIRST_INSTRUCTION [/INST]"
```
然后
>使用以下命令启动聊天。
>
>```shell
$ chmod +x chat.sh
$ ./chat.sh zh-models/7B/ggml-model-q4_0.gguf '请列举5条文明乘车的建议'
>```
>在提示符 `>` 之后输入你的prompt，`cmd/ctrl+c`中断输出，多行信息以`\`作为行尾。如需查看帮助和参数说明，请执行`./main -h`命令。下面介绍一些常用的参数：
>
>```
-c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
-f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt
-n 控制回复生成的最大长度（默认：128）
-b 控制batch size（默认：512）
-t 控制线程数量（默认：8），可适当增加
--repeat_penalty 控制生成回复中对重复文本的惩罚力度
--temp 温度系数，值越低回复的随机性越小，反之越大
--top_p, top_k 控制解码采样的相关参数
>```
>
>更详细的官方说明请参考：[https://github.com/ggerganov/llama.cpp/tree/master/examples/main](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)

### Step 4：架设server
此处的架设server的功能，是用于API调用、架设简易demo的，如果你希望自己架设服务器也是类似的原理。
>运行以下命令启动server，二进制文件`./server`在llama.cpp根目录，服务默认监听`127.0.0.1:8080`。这里指定模型路径、上下文窗口大小。如果需要使用GPU解码，也可指定`-ngl`参数。
>
>```shell
$ ./server -m ./zh-models/7B/ggml-model-q4_0.gguf -c 4096 -ngl 999
>```

这里的指令实际上用了两个相对路径，如果你希望把启动脚本放在非` llama.cpp`的其他目录下，那么将` ./server `替换为`xx/xx/xx/llama.cpp/server`即可，后面的`./zh-models/7B/ggml-model-q4_0.gguf`部分同理。此时可以得到如下脚本（abc.sh）文件内容：
```bash
/home/xxx/xxx/llama.cpp/server -m /home/xxx/xxx/llama.cpp/zh-models/7B/ggml-model-q4_0.gguf -c 4096 -ngl 999
```
执行该脚本（或命令）后即可启动服务：
>服务启动后，即可通过多种方式进行调用，例如利用`curl`命令。以下是一个示例脚本（同时存放在[scripts/llamacpp/server_curl_example.sh](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/llama-cpp/server_curl_example.sh)），将Alpaca-2的模板进行包装并利用`curl`命令进行API访问。
>
>```shell
># server_curl_example.sh
>
>SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
># SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
INSTRUCTION=$1
ALL_PROMPT="[INST] <<SYS>>\n$SYSTEM_PROMPT\n<</SYS>>\n\n$INSTRUCTION [/INST]"
CURL_DATA="{\"prompt\": \"$ALL_PROMPT\",\"n_predict\": 128}"
>
>curl --request POST \
>    --url http://localhost:8080/completion \
 >   --header "Content-Type: application/json" \
  >  --data "$CURL_DATA"
>```
>
>例如，我们给出一个示例指令。
>
>```shell
>$ bash server_curl_example.sh '请列举5条文明乘车的建议'
>```
>
>稍后返回响应结果。

返回的结果是json体的信息，其内容比较多，我们希望只得到文本反馈，那么就需要自己处理一下返回的内容并输出（这也是我没有贴原作者给出的.sh文件内容的原因）
通过分析原文件可以得知，我们需要包装curl命令并将自己的问题输入到服务器，然后get其回答即可，以下为示例代码：

# 待补充，包括7b文件的软链接

```python
import requests
import json
import re

```

# Part 2：本地部署GPT-SoVITS
_此内容依旧使用wsl2-Ubuntu2204进行部署，其他环境请参考[官方文档](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md)_

官方教程中对于Linux环境的解释较为简略，本文档旨在补充部分内容：
### Step 1:下载GPT-SoVITS源代码
![[Pasted image 20240303094229.png]]
如上图所示几种方法均可，本教程不教如何下载，现在假设你已经下载完成并将文件夹放在了你的~/目录下。

### Step 2:安装conda
可以根据conda的**[清华镜像源](https://link.zhihu.com/?target=https%3A//mirror.tuna.tsinghua.edu.cn/help/anaconda/)**去进行下载

```text
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

wget -c https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh        #清华的镜像源latest的版本的话就是说以后一直会更新最新的版本
```
上述命令得到的是.sh文件，使用如下命令安装：
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
具体过程不再赘述，可自行查阅
### Step 3:安装其他
在上述cunda安装完成后请重启命令行界面。
这里要求先开启cunda环境，以免造成GPT-SoVITS的配置影响其他软件运行
```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
```
此时你的命令行前应该会是这样的：
![[Pasted image 20240303101259.png]]
然后请进入你之前下载好的GPT-SoVITS文件夹内，如果此时使用`ls`命令，你可以在里面找到两个文件：`install.sh`和`requirements.txt`
此时运行指令，等待安装完成即可：
```bash
bash install.sh
```
（另：好像用`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`也可以安装，这里的回忆缺失了.jpg😭）
参考教程：[MAC教程 (yuque.com)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/znoph9dtetg437xb)
（对，我是看着MAC的教程安的）
下次再启动，只需要打开终端，定位到项目目录，进入conda环境，运行即可  
```bash
cd ~/GPT-SoVITS-main/GPT-SoVITS

conda activate GPTSoVits
```
### Step 4:推理
>在/GPT-SoVITS-main/路径下，运行以下命令即可启动webui界面：
>```bash
>python webui.py
>```

当然，你没有下载预训练模型肯定会报错，我是直接把win下整合包里面的东西丢到报错缺失的文件夹内的，你可以这样做：
>从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中。
>
>对于 UVR5（人声/伴奏分离和混响移除，附加），从 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 下载模型，并将它们放置在 `tools/uvr5/uvr5_weights` 中。

>中国地区用户可以进入以下链接并点击“下载副本”下载以上两个模型：
>
>- [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)
> 
>- [UVR5 Weights](https://www.icloud.com.cn/iclouddrive/0bekRKDiJXboFhbfm3lM2fVbA#UVR5_Weights)
>
对于中文自动语音识别（附加），从 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), 和 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 下载模型，并将它们放置在 `tools/damo_asr/models` 中。

~~我的目的是推理，又不是训练，而且都用命令行了还要什么ui界面🤪~~
基于种种原因（？）我决定找到推理界面直接运行推理即可
事实上，[有计划推出一个命令行一键训练和推理的脚本入口么 · Issue #489 · RVC-Boss/GPT-SoVITS (github.com)](https://github.com/RVC-Boss/GPT-SoVITS/issues/489)已经提出了疑问
顺藤摸瓜下，我得到了如下内容：
- 在GPT-SoVITS-main/GPT_SoVITS内存有二级界面的启动.py文件
- 推理界面的.py文件为`inference_webui.py`
- 该文件需要依赖GPT-SoVITS-main文件夹下的其他内容，并且作者将其写成了相对路径

理所当然的的就把他从GPT-SoVITS-main/GPT_SoVITS复制到了GPT-SoVITS-main下面。并且使用命令成功启动：
```bash
python inference_webui.py 
```

但是此时问题来了，我如果仿照Part1中用curl的方法推送并获取结果，服务器会报错：
```
{'detail': 'Method Not Allowed'}
```

很好，只能另寻他法。
然后我在GPT-SoVITS-main下翻到了api.py
谢谢你作者，为什么不把api调用方法写在文档里而是写在了文件里
（如果是因为我没找到api文档请接受我的道歉😭）

这个时候就简单了，直接启动远程端口：
```bash
python3 api.py
```
这里的端口号是9880，使用http://localhost:9880/即可访问。

这里同样使用curl方法推送参数并解析返回值，我已经写成了python文件如下：

# 待补充
```python
import requests
import json
```

# Part 3:联合使用Chinese-LLaMA-Alpaca-2和GPT-SoVITS
总结以上步骤即
- 在llama.cpp中使用bash start_sever.sh启动远程端口，
使用 pytohon3 inputques.py 读取question.txt内的问题，
将答案写入answer.txt（覆盖）和answers.txt（保存）。

- 在~/GPT-SoVITS-main中先使用conda activate GPTSoVits启动虚拟环境
再使用 python api.py启动远程端口，
使用python getvoice.py 读取~llama.cpp/answer.txt的内容并在~/下生成wav文件

大概就是这样💖
感谢您的阅读💕

~~爱来自Markdown~~
