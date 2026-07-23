from __future__ import annotations

import base64
import hashlib
import io
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import torch
import zstandard as zstd

COMMIT = "5b55d8ed55be4a9bc59550714d7f6a2e001f4136"
EXPECTED_SHA = "faf033b897c3ee115ac675898dc92b9f44dc2624ca0d6fc07e8183b4dbafa797"
NAMES = [
    "payload.00", "payload.01", "payload.02", "payload.03",
    "payload.040", "payload.041", "payload.042", "payload.043", "payload.05",
]
ROOT = f"https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/{COMMIT}/hf_clock_targeted_v3b/"
WORK = Path("/tmp/clock_targeted_v3")
SRC = WORK / "src"
CKPT = WORK / "checkpoints"
OUT = WORK / "out"
for path in (SRC, CKPT, OUT):
    path.mkdir(parents=True, exist_ok=True)

parts = []
for name in NAMES:
    chunk = urlopen(ROOT + name, timeout=90).read().strip()
    print("DOWNLOADED", name, len(chunk), flush=True)
    parts.append(chunk)
encoded = b"".join(parts)
blob = base64.b64decode(encoded, validate=True)
sha = hashlib.sha256(blob).hexdigest()
print("PAYLOAD_SHA256", sha, flush=True)
if sha != EXPECTED_SHA:
    raise RuntimeError(f"payload checksum mismatch: {sha} != {EXPECTED_SHA}")
data = zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob)).read()
with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
    archive.extractall(SRC)
PACKAGE = SRC / "targeted_v3_payload"
sys.path.insert(0, str(PACKAGE))
print("PACKAGE_READY", PACKAGE, flush=True)

from clock_program_hf.clockloop_core.train import train_one_seed, TrainConfig
from clock_program_hf.clockloop_core.data import ClockDataConfig
from clock_program_hf.clockloop_core.model import ClockLoopConfig
from clock_program_hf.regime_tagged_attention import RegimeTaggedConfig, run_regime_tagged_attention
from clock_program_hf.streaming_phase_v2 import StreamingV2Config, run_streaming_phase_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("TRAIN_DEVICE", device, flush=True)
train_cfg = TrainConfig(
    steps=1000,
    batch_size=64,
    learning_rate=1.2e-3,
    weight_decay=1e-4,
    grad_clip=1.0,
    aux_loss_weight=0.25,
    surprise_loss_weight=0.12,
    validation_every=150,
    train_k_values=(2, 4, 6, 8, 12),
    mixed_schedule_probability=0.55,
)
for seed in (11, 23, 37, 53, 71):
    checkpoint = CKPT / f"clockloop_seed_{seed}.pt"
    _, history = train_one_seed(
        seed,
        data_cfg=ClockDataConfig(),
        model_cfg=ClockLoopConfig(),
        train_cfg=train_cfg,
        checkpoint_path=checkpoint,
        device=device,
    )
    print("TRAINED", seed, history[-1], flush=True)

p14_dir = OUT / "p14_regime_tagged"
p16_dir = OUT / "p16_streaming_v2"
run_regime_tagged_attention(
    checkpoint_dir=CKPT,
    output_dir=p14_dir,
    device=str(device),
    config=RegimeTaggedConfig(batch_size=1024, detector_train_steps=700),
)
run_streaming_phase_v2(
    checkpoint_dir=CKPT,
    output_dir=p16_dir,
    device=str(device),
    config=StreamingV2Config(episodes=4096),
)
for name, path in (("P14", p14_dir / "summary.json"), ("P16", p16_dir / "summary.json")):
    print(f"FINAL_{name}_SUMMARY_JSON_BEGIN", flush=True)
    print(path.read_text(), flush=True)
    print(f"FINAL_{name}_SUMMARY_JSON_END", flush=True)
print("TARGETED_CLOCKLOOP_DONE", flush=True)
