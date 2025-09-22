from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_large",
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)
# pending...