"""I/O and immutable reference loading for the isolated benchmark."""
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "reference"
sys.dont_write_bytecode = True
REVISION = "A2_posthoc_screening_limits"
ERROR_LIMITS = {"e": .1, "p": .1, "h": .05}
A2_TASK_SHA256 = "aedbb18070ca981bdd1bc3e001458465ebed97f7ee8f08020f56217db897079a"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def verify_reference():
    manifest = json.loads((HERE / "reference_manifest.json").read_text())
    for rel, expected in manifest.items():
        actual = digest(SNAPSHOT / rel)
        if actual != expected:
            raise RuntimeError("reference hash mismatch: " + rel)
    return manifest


def reference():
    verify_reference()
    os.environ["FL_ROOT"] = str(SNAPSHOT)
    sys.path.insert(0, str(SNAPSHOT / "main_final2/revisions/p2v2_auction_side_20261022/exp"))
    import common_p2v2
    import p2v2_mech
    import rq2_timing_p2v2
    model = p2v2_mech.load("mfg_p2v2")
    return model, common_p2v2, rq2_timing_p2v2


def environment():
    import crypten
    import torch
    from omegaconf import OmegaConf
    cpu_info = Path("/proc/cpuinfo").read_text().splitlines()
    model_name = next((s.split(":", 1)[1].strip() for s in cpu_info if s.startswith("model name")), None)
    return dict(time=now(), host=socket.gethostname(), platform=platform.platform(), cpu_model=model_name,
                python=sys.version, executable=sys.executable, torch=torch.__version__,
                crypten=crypten.__version__, config=OmegaConf.to_container(crypten.cfg.config),
                cpu_count=os.cpu_count(), load=os.getloadavg(), threads_per_party=4,
                gloo_interface=os.environ.get("GLOO_SOCKET_IFNAME", "lo"))
