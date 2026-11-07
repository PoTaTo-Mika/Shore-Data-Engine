## Shore-Data-Engine

面向语音数据集的一体化流水线：各站点爬虫保持原样产出原始音频，`crawler.pack` 统一转码 Opus 并按固定体积打成 tar，`engine.run` 按配置拉起服务，对 tar 内音频串行执行 stage1/2/3，最终流式写出 jsonl 标注。

### 设计目标

- 爬虫只负责下载；格式与打包统一在 pack 阶段收敛，不改动各站点脚本。
- 每个 stage 的模型可配置，`stages[].model` 指向 `models` 表中的条目。
- stage1 归为 audio preprocess，当前实现是 UVR（Demucs 人声分离）；去混响等模型待引入后并入这一阶段。
- stage2 做 VAD 切分，结果写回同一个 tar。
- stage3 通过配置启动的后端服务发请求做 ASR，并维护 jsonl 作为数据映射。
- 后续可能增加 stage4/5/6，但当前不提前抽象插件或注册机制，先把标准三件套做实。

### 流程

```text
站点爬虫下载原始音频 (mp3/m4a/wav/...)
  → crawler.pack 统一转码 opus，按 opus 产物真实体积打 tar
  → engine.run 按 services 声明顺序拉起服务并等待健康检查
  → 每个 tar 串行执行 stage1 UVR / stage2 Slicer / stage3 ASR
  → data/annotations.jsonl
```

### 数据契约

tar 由 `crawler.pack` 产出，放在 `data/` 根目录：

- 命名为 `<桶路径>_<YYYYmmdd_HHMMSS>_<序号>.tar`，例如 `en_album_20261106_203000_000.tar`。
- 成员路径保留桶目录结构，如 `en/album/clip.opus`；音频统一为 opus，非音频成员原样保留。
- 打包只收 `.opus`，成功后删除 opus 源；原始文件与 `.xxx.mp3.done` 标记默认保留。
- 引擎 stage1/stage2 会原子重写 tar，并写入 `_shore.json`：`{"stages": ["stage1", "stage2"]}`，重跑时自动跳过已完成阶段。
- stage2 的切片命名为 `<stem>_000.opus`、`<stem>_001.opus`……

标注结果 `data/annotations.jsonl`，每行一条：

```json
{"tar_name": "en_album_20261106_203000_000.tar", "audio_id": "en/album/clip_000.opus", "asr_text": "..."}
```

`audio_id` 是 tar 内成员路径（切分后的片段带序号）。后续新增特征只需在 `data_process.annotation.Annotation` 加字段；已写出的 `(tar_name, audio_id)` 在重跑时跳过。

### 目录结构

```text
Shore-Data-Engine/
  crawler/
    pack.py                # 统一转码、稳定性判定、按体积打 tar
    upload.py              # 可选：上传 tar 到 S3
    <站点>/crawler.py      # 各站点原始爬虫，无需改动
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
  third-party/demucs/      # UVR 依赖（git submodule）
  tests/
  data/                    # 运行期数据：桶目录、*.tar、annotations.jsonl
```

### 环境要求与安装

- Python 3.10+；必须安装 `ffmpeg`（含 `libopus`），转码、UVR 与切分都依赖它。
- UVR 需要 `torch` / `demucs`，切分需要 `numpy` / `soundfile` / `librosa`；部分站点爬虫需要 `requests` / `beautifulsoup4` / `aiohttp` / `boto3` 与 Node.js。

```bash
git submodule update --init
python -m pip install -e third-party/demucs
python -m pip install -e ".[all]"
```

### 快速开始

1. 采集数据。各站点爬虫保持原样、按各自方式运行，产出原始格式即可。

```bash
python -m crawler.qtfm.free_crawler
python -m crawler.spotify.free_crawler
```

外部工具直接产出文件（BBDown、ffmpeg 拉流等）同样放进 `data/`，pack 会一并处理。

2. 统一转码并打 tar。判定文件写完靠「mtime 早于 `SHORE_STABLE_SECONDS` + 连续两轮 size/mtime 不变」，转码由 `SHORE_TRANSCODE_WORKERS` 个 worker 并行执行；失败不写标记，下一轮自动重试。分批按 **opus 产物的真实体积**计算，达到 `SHORE_TAR_MAX_SIZE_GB`（默认 1 GiB）即打包，滞留超过 `SHORE_STALE_SECONDS`（默认 3600s）强制打包。

```bash
python -m crawler.pack             # 常驻监控
python -m crawler.pack --once      # 只跑一轮
```

默认保留原始文件，这样爬虫重跑时「已存在就跳过」的判断仍然生效；`SHORE_DELETE_RAW=1` 可在转码后直接删掉原始文件。可选 `python -m crawler.upload` 上传到 S3。

3. 运行标注引擎。从仓库根目录执行：

```bash
python -m engine.run --dry-run -c engine/configs/run.yaml   # 查看 stage 顺序与 tar 数量
python -m engine.run -c engine/configs/run.yaml             # 完整串行执行
python -m engine.run --stage stage3 --limit 10             # 只跑 ASR，限制条数
```

`pip install -e .` 之后等价入口是 `shore`（`shore --dry-run` 等）。**不要**用 `python engine/run.py`：以文件路径运行会让包导入失效并抛 `ModuleNotFoundError`。

### 配置说明

`engine/configs/run.yaml` 是唯一配置源，由 `load_config` 实际读取：

- `data_root` / `output_name` / `audio_extensions`：数据根目录、jsonl 文件名、参与处理的音频扩展名。
- `services`：按声明顺序启动；`command` 可选，删除即接管已运行的服务；另有 `base_url`、`health_path`、`startup_timeout_s`、`timeout_s`、`retry`。
- `models`：模型表。stage 通过 `model` 引用表中的键；`endpoint` 支持 `/v1/audio/transcriptions`（multipart）与 `/v1/chat/completions`（base64）。
- `stages`：`stage1` / `stage2` / `stage3` 对应 `uvr` / `slicer` / `asr`，顺序即执行顺序；ASR stage 额外用 `service` 指定后端。

`${VAR:-default}` 在加载时展开，例如 `OMNI_BASE_URL`、`OMNI_API_KEY`。pack 侧另有 `SHORE_DATA_DIR`、`SHORE_OPUS_BITRATE`、`SHORE_STABLE_SECONDS`、`SHORE_TRANSCODE_WORKERS`、`SHORE_DELETE_RAW`、`SHORE_TAR_MAX_SIZE_GB`、`SHORE_STALE_SECONDS`、`SHORE_SCAN_INTERVAL`；上传侧为 `S3_ENDPOINT`、`S3_KEY`、`S3_SECRET`、`S3_BUCKET`、`S3_PREFIX`。

### 验证与常见问题

```bash
python -m pytest -q
python -m ruff check crawler/pack.py engine/run.py
python -m engine.run --dry-run
```

- 转码失败：确认 `ffmpeg -encoders | grep libopus` 可用；失败文件会保留并重试。想重新转码（例如改了码率）删掉对应的 `.done` 标记。
- 磁盘占用：默认同时保留 raw 与 opus，约翻倍；一次性爬取可用 `SHORE_DELETE_RAW=1`。
- 服务未就绪：`startup_timeout_s` 内 `/health` 未返回 200 会退出，可手工启动服务并移除 `command`。
- 幂等：tar 内 `_shore.json` 记录已完成阶段，jsonl 记录已标注成员，两者共同保证重跑不重复处理。

### 许可证

本项目遵循仓库根目录的 `LICENSE`；`data_process/slicer/rms_vad.py` 中的切分逻辑来自 openvpi/audio-slicer（MIT）。
