## Shore-Data-Engine

面向语音数据集的一体化流水线：各站点爬虫保持原样产出原始音频，由 `crawler.pack` 统一转码 Opus 并按固定数据量打成 tar，标注引擎直接读写 tar 内的音频，串行完成 UVR、切分、ASR，并流式输出 jsonl 标注。

### 流程

```text
各站点爬虫下载原始音频 (mp3/m4a/wav/...，代码不改)
  → crawler.pack 统一转码 opus + 按 opus 产物真实体积打 tar
  → engine.run 按配置顺序启动服务
  → 每个 tar 串行执行 stage1 UVR / stage2 Slicer / stage3 ASR
  → data/annotations.jsonl
```

### 目录结构

```text
Shore-Data-Engine/
  crawler/
    pack.py                # 统一转码、稳定性判定、按体积打 tar
    upload.py              # 可选：上传 tar 到 S3
    <站点>/crawler.py      # 各站点原始爬虫，无需任何改动
  data_process/
    data_reader.py         # tar 内音频的流式读写与原子重写
    annotation.py          # jsonl 标注记录与断点续跑
    stages.py              # UVR / 切分的字节级算子
    uvr/, slicer/          # 底层算子实现（Demucs / RMS VAD）
  engine/
    run.py                 # 配置驱动的串行管线入口
    backends.py            # 服务生命周期与 ASR 请求
    configs/run.yaml       # 服务、模型、stage1/2/3 的唯一配置源
    sglang-omni/           # 模型服务子模块
  tests/
  data/                    # 运行期数据：桶目录、*.tar、annotations.jsonl
  logs/
```

### 环境要求

- Python 3.10+（sglang-omni 服务端建议 3.11 / 3.12）
- 必须安装 `ffmpeg`（含 `libopus`）：转码、UVR 与切分都依赖它
- UVR 需要 `torch` / `demucs`，切分需要 `numpy` / `soundfile` / `librosa`
- 模型服务按 `engine/configs/run.yaml` 中的 `command` 拉起；不需要引擎拉起时删除该字段并自行启动服务
- 爬虫额外依赖 `requests` / `beautifulsoup4` / `aiohttp` / `boto3`，部分站点需要 Node.js

### 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
git submodule update --init
pip install -e third-party/demucs
pip install -e ".[all]"
```

### 快速开始

1. 采集数据。各站点爬虫保持原样、按各自方式运行，产出原始格式即可，不需要任何改动：

```bash
python -m crawler.qtfm.free_crawler
python -m crawler.spotify.free_crawler
```

外部工具直接产出文件（BBDown、ffmpeg 拉流等）同样直接放进 `data/`，pack 会一并处理。

2. 统一转码并打 tar。pack 递归扫描 `data/`，等文件写完后并行转成 opus，再按 **opus 产物的真实体积**分批；桶是任意层级的目录，达到 `SHORE_TAR_MAX_SIZE_GB`（默认 1 GiB）即打包，滞留超过 `SHORE_STALE_SECONDS`（默认 3600s）强制打包：

```bash
python -m crawler.pack             # 常驻监控
python -m crawler.pack --once      # 只跑一轮
```

判定文件写完靠"mtime 早于 `SHORE_STABLE_SECONDS` + 连续两轮 size/mtime 不变"，转码由 `SHORE_TRANSCODE_WORKERS`（默认 4）个 worker 并行执行，失败会保留源文件并在下一轮重试。

产物 `xxx.opus` 与原始文件同目录，同时写隐藏标记 `.xxx.mp3.done` 避免重复转码。**默认保留原始文件**，这样爬虫重跑时"已存在就跳过"的判断仍然生效；打包只收 opus，成功后删除 opus，原始文件与标记保留，确认不再需要重跑爬虫后可手动清理。设 `SHORE_DELETE_RAW=1` 可在转码后直接删掉原始文件（此时重跑爬虫会重新下载）。

tar 命名为 `data/<桶名>_<时间戳>_<序号>.tar`，成员路径保留桶目录结构。可选 `python -m crawler.upload` 上传到 S3，需设置 `S3_ENDPOINT`、`S3_KEY`、`S3_SECRET`、`S3_BUCKET`。

3. 运行标注引擎：

```bash
shore --dry-run                                  # 查看 stage 顺序与 tar 数量
shore --config engine/configs/run.yaml            # 完整串行执行
shore --stage stage3 --limit 10                  # 只跑 ASR，限制条数
```

引擎按 `services` 顺序拉起服务并等待 `/health`，然后对每个 tar 依次执行 `stage1 UVR → stage2 Slicer → stage3 ASR`。前两个阶段把结果直接写回 tar（manifest 记录已完成阶段，重跑自动跳过），ASR 阶段流式读取 tar 成员并追加写 jsonl。

4. 标注结果 `data/annotations.jsonl`，每行一条：

```json
{"tar_name": "en_20261106_120000_000.tar", "audio_id": "album/clip_000.opus", "asr_text": "..."}
```

`audio_id` 是 tar 内成员路径（切分后的片段带序号）。后续引入新特征时，在 `data_process.annotation.Annotation` 增加字段即可，无需改动读写框架。已写出的 `(tar_name, audio_id)` 会在重跑时跳过。

### 配置说明

`engine/configs/run.yaml` 是唯一配置源：

- `data_root` / `output_name` / `audio_extensions`：数据根目录、jsonl 文件名、音频扩展名
- `services`：按声明顺序启动，支持 `command`、`base_url`、`health_path`、`startup_timeout_s`、`retry`
- `models`：模型参数，`endpoint` 支持 `/v1/audio/transcriptions`（multipart）与 `/v1/chat/completions`（base64）
- `stages`：`stage1` / `stage2` / `stage3` 对应 `uvr` / `slicer` / `asr`，顺序即执行顺序

环境变量 `${VAR:-default}` 会在加载时展开，例如 `OMNI_BASE_URL`、`OMNI_API_KEY`。

pack 侧另有独立的环境变量：`SHORE_DATA_DIR`（默认 `data`）、`SHORE_STABLE_SECONDS`、`SHORE_TRANSCODE_WORKERS`、`SHORE_OPUS_BITRATE`、`SHORE_DELETE_RAW`、`SHORE_TAR_MAX_SIZE_GB`、`SHORE_STALE_SECONDS`、`SHORE_SCAN_INTERVAL`。

### 常见问题

- 转码失败：确认 `ffmpeg -encoders | grep libopus` 可用；失败的文件不会写标记，会保留并在下一轮自动重试
- 想重新转码：删掉对应的 `.xxx.mp3.done` 标记（如改了 `SHORE_OPUS_BITRATE`）
- 磁盘占用：默认同时保留 raw 与 opus，约翻倍；一次性爬取可用 `SHORE_DELETE_RAW=1`
- 服务未就绪：`startup_timeout_s` 内 `/health` 未返回 200 会报错退出，可先手工启动服务并移除 `command`
- 重跑重复处理：tar 内 `_shore.json` 记录已完成阶段，jsonl 记录已标注成员，两者共同保证幂等
- 首次运行 UVR 会下载 Demucs 权重，切分 / ASR 模型按各自配置加载

### 许可证

本项目遵循仓库根目录的 `LICENSE`；`data_process/slicer/rms_vad.py` 中的切分逻辑来自 openvpi/audio-slicer（MIT）。
