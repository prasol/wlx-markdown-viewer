from __future__ import annotations

import base64
import hashlib
import io
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import zstandard as zstd

COMMIT = "4e419a9ce0023e8f41160c14c401e12bea7c76b2"
EXPECTED_SHA = "faf033b897c3ee115ac675898dc92b9f44dc2624ca0d6fc07e8183b4dbafa797"
NAMES = [
    "payload.00", "payload.01", "payload.02", "payload.03",
    "payload.040", "payload.041", "payload.042", "payload.043", "payload.05",
]
ROOT = f"https://raw.githubusercontent.com/prasol/wlx-markdown-viewer/{COMMIT}/hf_clock_targeted_v3b/"
WORK = Path("/tmp/readout_null_v3")
SRC = WORK / "src"
OUT = WORK / "out"
SRC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

encoded = b"".join(urlopen(ROOT + name, timeout=90).read().strip() for name in NAMES)
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

from clock_program_hf.readout_null import ReadoutNullConfig, run_readout_null

output_dir = OUT / "p15_readout_null"
run_readout_null(
    output_dir=output_dir,
    device="cuda",
    config=ReadoutNullConfig(max_examples=12),
)
summary = output_dir / "summary.json"
print("FINAL_P15_SUMMARY_JSON_BEGIN", flush=True)
print(summary.read_text(), flush=True)
print("FINAL_P15_SUMMARY_JSON_END", flush=True)
print("READOUT_NULL_DONE", flush=True)
