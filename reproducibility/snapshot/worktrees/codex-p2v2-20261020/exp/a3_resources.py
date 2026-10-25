"""Resource admission for one node, including CUDA processes outside P2."""
import os
from pathlib import Path
import subprocess

MIB = 1024 ** 2
RESERVE_MIB = 3 * 1024


def resources():
    fields = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True).splitlines()[0].split(",")
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    mem = {p[0].rstrip(":"): int(p[1]) for p in (line.split() for line in Path("/proc/meminfo").read_text().splitlines())}
    return {"free_mib": int(fields[0]), "used_mib": int(fields[1]), "gpu_util": int(fields[2]),
            "pswpin": int(vm["pswpin"]), "pswpout": int(vm["pswpout"]),
            "ram_available_mib": mem["MemAvailable"] / 1024,
            "swap_used_mib": (mem["SwapTotal"] - mem["SwapFree"]) / 1024,
            "logical_cpus": os.cpu_count(), "load": list(os.getloadavg())}


def can_start(snapshot, unobserved_reservations=0, peak_mib=1536):
    return (snapshot["free_mib"] - peak_mib * (1 + unobserved_reservations) >= RESERVE_MIB
            and snapshot["ram_available_mib"] >= 4096)


def token(pid):
    try:
        v = (Path("/proc") / str(pid) / "stat").read_text().split()
        return v[21] if v[2] != "Z" else None
    except (OSError, IndexError):
        return None


def gpu_pids():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return {int(p.strip()) for p in rows.splitlines() if p.strip().isdigit()}


def environment():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
