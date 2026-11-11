"""Stage dispatch for VAD slicing: `python -m data_process.vad.dispatch`."""

from data_process.protocol import OperatorBackend

BACKEND = OperatorBackend(
    module="data_process.vad.dispatch",
    endpoint="/v1/audio/segments",
    operators={
        "rms_vad": "data_process.vad.rms",
        "silero_vad": "data_process.vad.silero",
        "ten_vad": "data_process.vad.ten",
    },
)

if __name__ == "__main__":
    BACKEND.serve()
