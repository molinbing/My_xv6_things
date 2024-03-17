# 部署 whisper.cpp 并实现自动转化 16kHz.wav 文件输出和保存，测试流式语音的转化

_本教程使用 wsl2-Ubuntu2204 进行部署,其他环境请参考[官方文档](https://github.com/ggerganov/whisper.cpp)和 py 版：[官方文档](https://github.com/openai/whisper)_

## 一、基础部署

首先克隆存储库：

```
git clone https://github.com/ggerganov/whisper.cpp.git
```

然后，下载以 ggml 格式转换的 Whisper 模型之一。例如：

```
bash ./models/download-ggml-model.sh base.en
```

> _注：本教程使用的默认模型是 base 模型，来自：https://github.com/ggerganov/whisper.cpp/tree/master/models。 请自行下载模型放入 models/或者按照后续提示下修改.sh 文件内容。_

使用 NVIDIA 卡，可以通过 cuBLAS 和自定义 CUDA 内核在 GPU 上高效地处理模型。 首先，确保您已安装： https://developer.nvidia.com/cuda-downloadscuda

现在使用 cuBLAS 支持进行构建：whisper.cpp

```
make clean
WHISPER_CUBLAS=1 make -j
```

> 如果没有 N 卡，请直接按照如下步骤尝试，如有则跳过
>
> 现在构建主示例并转录一个音频文件，如下所示：
>
> ```
> # build the main example
> make
>
> # transcribe an audio file
> ./main -f samples/jfk.wav
> ```

然后安装 ffmpeg

```
$ sudo apt install ffmpeg
```

在`/xxx/whisper`目录下创建如下`你自己起的名字.sh`文件

```
#!/bin/bash

# 获取输入音频文件路径
audio_path=$1

# 检查输入文件是否存在
if [ ! -f "$audio_path" ]; then
  echo "输入文件不存在: $audio_path"
  exit 1
fi

# 获取输出文件名
output_name=${audio_path%.*}_16.wav

# 转换音频格式
ffmpeg -i "$audio_path" -ac 1 -ar 16000 "$output_name"

# 打印输出文件路径
echo "输出文件: $output_name"

# 语音转化
./main -m models/base.bin -otxt -l zh -f $output_name
```

然后请在`/xxx/whisper/samples/`下放入你需要转录为文本的音频文件`音频文件.wav`

通过如下方式运行

```
~/xxx/whisper.cpp$ bash 你自己起的名字.sh ./samples/音频文件.wav
```

至此你应该会看到类似如下的输出：

```
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Guessed Channel Layout for Input Stream #0.0 : mono
Input #0, wav, from './samples/2.wav':
  Duration: 00:00:16.24, bitrate: 705 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to './samples/2_16.wav':
  Metadata:
    ISFT            : Lavf58.76.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
    Metadata:
      encoder         : Lavc58.134.100 pcm_s16le
size=     508kB time=00:00:16.23 bitrate= 256.1kbits/s speed=1.15e+03x
video:0kB audio:507kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.015011%
输出文件: ./samples/1_16.wav
whisper_init_from_file_with_params_no_state: loading model from 'models/base.bin'
whisper_model_load: loading model
whisper_model_load: n_vocab       = 51865
whisper_model_load: n_audio_ctx   = 1500
whisper_model_load: n_audio_state = 512
whisper_model_load: n_audio_head  = 8
whisper_model_load: n_audio_layer = 6
whisper_model_load: n_text_ctx    = 448
whisper_model_load: n_text_state  = 512
whisper_model_load: n_text_head   = 8
whisper_model_load: n_text_layer  = 6
whisper_model_load: n_mels        = 80
whisper_model_load: ftype         = 1
whisper_model_load: qntvr         = 0
whisper_model_load: type          = 2 (base)
whisper_model_load: adding 1608 extra tokens
whisper_model_load: n_langs       = 99
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX xxxx GPU, compute capability 8.6, VMM: yes
whisper_backend_init: using CUDA backend
whisper_model_load:    CUDA0 total size =   147.37 MB
whisper_model_load: model size    =  147.37 MB
whisper_backend_init: using CUDA backend
whisper_init_state: kv self size  =   16.52 MB
whisper_init_state: kv cross size =   18.43 MB
whisper_init_state: compute buffer (conv)   =   16.39 MB
whisper_init_state: compute buffer (encode) =  132.07 MB
whisper_init_state: compute buffer (cross)  =    4.78 MB
whisper_init_state: compute buffer (decode) =   96.48 MB

system_info: n_threads = 4 / 16 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | METAL = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | CUDA = 1 | COREML = 0 | OPENVINO = 0

main: processing './samples/1_16.wav' (259808 samples, 16.2 sec), 4 threads, 1 processors, 5 beams + best of 5, lang = zh, task = transcribe, timestamps = 1 ...


[00:00:00.000 --> 00:00:02.840]  输出文本
[00:00:02.840 --> 00:00:05.720]  输出文本
[00:00:05.720 --> 00:00:08.760]  输出文本
[00:00:08.760 --> 00:00:10.960]  输出文本
[00:00:10.960 --> 00:00:12.800]  输出文本
[00:00:12.800 --> 00:00:15.720]  输出文本

output_txt: saving output to './samples/1_16.wav.txt'

whisper_print_timings:     load time =  3844.00 ms
whisper_print_timings:     fallbacks =   0 p /   0 h
whisper_print_timings:      mel time =    26.28 ms
whisper_print_timings:   sample time =   201.05 ms /   334 runs (    0.60 ms per run)
whisper_print_timings:   encode time =    42.10 ms /     1 runs (   42.10 ms per run)
whisper_print_timings:   decode time =     0.00 ms /     1 runs (    0.00 ms per run)
whisper_print_timings:   batchd time =   423.16 ms /   332 runs (    1.27 ms per run)
whisper_print_timings:   prompt time =     0.00 ms /     1 runs (    0.00 ms per run)
whisper_print_timings:    total time =  4550.90 ms
```

至此你可能输出成功或者失败，请按照如下说明修正：

- 请参阅输入：`./main -h` 后得到的列表。

- 输出文本在音频同文件夹下的.txt 文件内，通过-otxt 参数控制输出，暂未找到指定文件的方法。

- 输出失败；模型错误；输出英文：请在 https://github.com/ggerganov/whisper.cpp/tree/master/models 下载本文章使用的 base 模型。或者将.sh 文件中的语句如下修改：

```
#原文
./main -m models/base.bin -otxt -l zh -f $output_name
#修改
./main -m models/你使用的模型全名 -otxt -l 你使用的语言类型 -f $output_name
```

> 这是可能的语言支持列表：
> LANGUAGES = {
> "en": "english",
> "zh": "chinese",
> "de": "german",
> "es": "spanish",
> "ru": "russian",
> "ko": "korean",
> "fr": "french",
> "ja": "japanese",
> "pt": "portuguese",
> "tr": "turkish",
> "pl": "polish",
> "ca": "catalan",
> "nl": "dutch",
> "ar": "arabic",
> "sv": "swedish",
> "it": "italian",
> "id": "indonesian",
> "hi": "hindi",
> "fi": "finnish",
> "vi": "vietnamese",
> "he": "hebrew",
> "uk": "ukrainian",
> "el": "greek",
> "ms": "malay",
> "cs": "czech",
> "ro": "romanian",
> "da": "danish",
> "hu": "hungarian",
> "ta": "tamil",
> "no": "norwegian",
> "th": "thai",
> "ur": "urdu",
> "hr": "croatian",
> "bg": "bulgarian",
> "lt": "lithuanian",
> "la": "latin",
> "mi": "maori",
> "ml": "malayalam",
> "cy": "welsh",
> "sk": "slovak",
> "te": "telugu",
> "fa": "persian",
> "lv": "latvian",
> "bn": "bengali",
> "sr": "serbian",
> "az": "azerbaijani",
> "sl": "slovenian",
> "kn": "kannada",
> "et": "estonian",
> "mk": "macedonian",
> "br": "breton",
> "eu": "basque",
> "is": "icelandic",
> "hy": "armenian",
> "ne": "nepali",
> "mn": "mongolian",
> "bs": "bosnian",
> "kk": "kazakh",
> "sq": "albanian",
> "sw": "swahili",
> "gl": "galician",
> "mr": "marathi",
> "pa": "punjabi",
> "si": "sinhala",
> "km": "khmer",
> "sn": "shona",
> "yo": "yoruba",
> "so": "somali",
> "af": "afrikaans",
> "oc": "occitan",
> "ka": "georgian",
> "be": "belarusian",
> "tg": "tajik",
> "sd": "sindhi",
> "gu": "gujarati",
> "am": "amharic",
> "yi": "yiddish",
> "lo": "lao",
> "uz": "uzbek",
> "fo": "faroese",
> "ht": "haitian creole",
> "ps": "pashto",
> "tk": "turkmen",
> "nn": "nynorsk",
> "mt": "maltese",
> "sa": "sanskrit",
> "lb": "luxembourgish",
> "my": "myanmar",
> "bo": "tibetan",
> "tl": "tagalog",
> "mg": "malagasy",
> "as": "assamese",
> "tt": "tatar",
> "haw": "hawaiian",
> "ln": "lingala",
> "ha": "hausa",
> "ba": "bashkir",
> "jw": "javanese",
> "su": "sundanese",
> "yue": "cantonese",
> }

## 二、测试流式语音的转化

_本部分参考[流式推理](https://github.com/ggerganov/whisper.cpp/tree/master/examples/stream)部分测试，仅记录作用，成功性不做保证。_

该工具依赖于 SDL2 库来捕获来自麦克风的音频。

```
# Install SDL2 on Linux
sudo apt-get install libsdl2-dev
```

请重新构建

```
make clean
make stream
# 可选选项，未知效果
# make stream WHISPER_CUBLAS=1 -j
```

请确保在项目根目录下构建，否则会出现如下错误：

```
whisper.cpp/examples/stream$ make stream
g++     stream.cpp   -o stream
stream.cpp:6:10: fatal error: common/sdl.h: No such file or directory
    6 | #include "common/sdl.h"
      |          ^~~~~~~~~~~~~~
compilation terminated.
make: *** [<builtin>: stream] Error 1
```

已知本流式构建和原构建二者不能共存，使用相同脚本启动方法会出现如下内容：

```
$ bash turn16_output.sh ./samples/2.wav
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared
  libavutil      56. 70.100 / 56. 70.100
  libavcodec     58.134.100 / 58.134.100
  libavformat    58. 76.100 / 58. 76.100
  libavdevice    58. 13.100 / 58. 13.100
  libavfilter     7.110.100 /  7.110.100
  libswscale      5.  9.100 /  5.  9.100
  libswresample   3.  9.100 /  3.  9.100
  libpostproc    55.  9.100 / 55.  9.100
Guessed Channel Layout for Input Stream #0.0 : mono
Input #0, wav, from './samples/2.wav':
  Duration: 00:00:16.24, bitrate: 705 kb/s
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 44100 Hz, mono, s16, 705 kb/s
File './samples/2_16.wav' already exists. Overwrite? [y/N] y
Stream mapping:
  Stream #0:0 -> #0:0 (pcm_s16le (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, wav, to './samples/2_16.wav':
  Metadata:
    ISFT            : Lavf58.76.100
  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s
    Metadata:
      encoder         : Lavc58.134.100 pcm_s16le
size=     508kB time=00:00:16.23 bitrate= 256.1kbits/s speed= 695x
video:0kB audio:507kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.015011%
输出文件: ./samples/2_16.wav
turn16_output.sh: line 22: ./main: No such file or directory
```

硬件限制暂未继续测试，猜测本构建需要使用如下方法启动：

```
#工具每半秒对音频进行一次采样
./stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length 5000
```

设置参数`--step`为 0 以启用滑动窗口模式：

```
 ./stream -m ./models/ggml-small.en.bin -t 6 --step 0 --length 30000 -vth 0.6
```

在这种模式下，工具只有在检测到某些语音活动之后才会进行转录。一个非常基本的 VAD 检测器被使用，但在理论上一个更复杂的方法可以添加。`-Vth` 参数确定 VAD 阈值-更高的值将使其更频繁地检测静默。最好是根据特定的用例对其进行调优，但是一般来说，大约`0.6`的值应该是可以的。当检测到静音时，它将转录最后`--length`为毫秒的音频，并输出一个适合解析的转录块。

官方也给出了更多的语言直接输入的示例：

[接受来自麦克风的语音命令](https://github.com/ggerganov/whisper.cpp/tree/725350d4ea1545d890fe41f815b851cbc57838f6/examples/command)

[按下以结束转录](https://github.com/ggerganov/whisper.cpp/tree/725350d4ea1545d890fe41f815b851cbc57838f6/examples/whisper.nvim)

[talk-llama](https://github.com/ggerganov/whisper.cpp/tree/master/examples/talk-llama#talk-llama)
