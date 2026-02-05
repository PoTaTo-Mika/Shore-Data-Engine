# 先安装主要内容
pip install -r requirements.txt
pip install vllm

# 8卡机器直接用这个编译
MAX_JOBS=32 pip install -U flash-attn --no-build-isolation