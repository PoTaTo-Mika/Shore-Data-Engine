"""Stage dispatch for vocal isolation: `python -m data_process.uvr.dispatch`."""

from data_process.protocol import OperatorBackend

BACKEND = OperatorBackend(
    module="data_process.uvr.dispatch",
    endpoint="/v1/audio/separations",
    operators={
        "demucs": "data_process.uvr.demucs",
        "mdx_net": "data_process.uvr.mdx",
    },
)

if __name__ == "__main__":
    BACKEND.serve()
