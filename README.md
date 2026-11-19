Agent不要改这个README.md

核心设计思路：仅通过更改配置文件来设置全链路流水线处理的模型和参数。

每个stage都会有一个总调度文件。

然后engine当中run会统一organize这些调度文件。

所有的stage都采用backend+请求响应的方式进行数据处理。

就比如说我们的爬虫爬下来的都是原始的wav等，然后在第0阶段，外部打包好tar数据，然后直接对tar进行读写，使用jsonl做标注数据管理，从而隔离开复杂的处理中间态。

然后jsonl是统一的，把各种标注写入同一个jsonl当中。

最终数据规约：

tar_name -> jsonl_name，完全匹配

tar:

name -> audio

jsonl:

{
    "name": "file_name_{slice}",
    "asr_text": "This is transcription",
    "any other features": xxxx
}

工作流程是，把每个阶段设置为一个函数：

uvr(waveform)，然后这里uvr是一个函数函数，本身具备一个模型属性，根据传入的模型不同决定。