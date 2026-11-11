Agent不要改这个README.md

核心设计思路：仅通过更改配置文件来设置全链路流水线处理的模型和参数。

每个stage都会有一个总调度文件。

然后engine当中run会统一organize这些调度文件。

所有的stage都采用backend+请求响应的方式进行数据处理。