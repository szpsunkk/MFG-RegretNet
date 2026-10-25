"""One resource-aware worker per machine; numerical executor is unchanged."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT, WORK, OUT, atomic_json, digest
from p2_state import check_identity, read, safe_progress
from a3_resources import resources, can_start, gpu_pids, token, environment, RESERVE_MIB


def assigned(job, machine):
    if not job["gpu"]:
        return machine == "101"
    if job["phase"].startswith("smoke_"):
        return machine == "101"
    stage = job["phase"].split("_")[0]
    table = OUT / "assignments" / (stage + ".json")
    if not table.exists():
        return False
    data = read(table)
    from a3_campaign import cell_key
    key = cell_key(job["cell"])
    if data["environment_sha256"].get(machine) != digest(OUT / "operations" / machine / "environment.json"):
        raise RuntimeError("worker environment differs from fixed block assignment")
    return data["cells"].get(key) == machine


def sharing_source(job, queue, machine):
    proof = OUT / "operations" / machine / "shared_validation.json"
    if not proof.exists() or not read(proof)["enabled"]:
        return None
    # Only within the same stage, cell, seed, and assigned environment.
    for other in queue:
        if other["id"] == job["id"] or not other["gpu"] or not assigned(other, machine):
            continue
        if (other.get("fl_fingerprint") == job.get("fl_fingerprint") and
            other["phase"].split("_")[0] == job["phase"].split("_")[0] and
            other["cell"] == job["cell"] and other["seed"] == job["seed"] and Path(other["done"]).exists()):
            return other["out"]
    return None


def run(machine):
    status = OUT / "workers" / (machine + ".json")
    status.parent.mkdir(parents=True, exist_ok=True)
    lock = status.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    live, failures = {}, []
    start_swap = resources()
    while True:
        check_identity()
        queue_path = OUT / "queue.json"
        queue = read(queue_path)["jobs"] if queue_path.exists() else []
        queue.sort(key=lambda j: (j["priority"], j["id"]))
        probe = OUT / "operations" / machine / "concurrency.json"
        validated = OUT / "operations" / machine / "PREFLIGHT_DONE.json"
        limit = read(probe)["selected_k"] if probe.exists() and validated.exists() else 0
        if machine == "local":
            reg = OUT / "operations/local/regression.json"
            if not reg.exists() or not read(reg)["local_eligible"]:
                limit = 0
            # Two slots remain reserved for LN, even while its producer is absent.
            limit = max(0, limit - 2)
        # Reattach after an operations-process restart without duplicating work.
        for job in queue:
            if not assigned(job, machine) or Path(job["done"]).exists():
                continue
            p = Path(job["out"]) / "process.json"
            if p.exists():
                previous = read(p)
                pid = previous["pid"]
                if pid not in live and token(pid) == previous["start_token"]:
                    live[pid] = (None, None, job, previous["start_token"])
        for pid, (proc, log, job, tok) in list(live.items()):
            alive = proc.poll() is None if proc is not None else token(pid) == tok
            if alive:
                continue
            if log:
                log.close()
            del live[pid]
            if not Path(job["done"]).exists() or (proc is not None and proc.returncode):
                failure = {"id": job["id"], "out": job["out"], "returncode": None if proc is None else proc.returncode}
                failures.append(failure)
                atomic_json(OUT / "workers" / (machine + "_failure.json"), failure)
        snapshot = resources()
        no_swap = snapshot["pswpin"] == start_swap["pswpin"] and snapshot["pswpout"] == start_swap["pswpout"]
        gpu_count = sum(x[2]["gpu"] for x in live.values())
        cpu_count = len(live) - gpu_count
        observed_gpu = gpu_pids()
        sharing_proof = OUT / "operations" / machine / "shared_validation.json"
        sharing_enabled = sharing_proof.exists() and read(sharing_proof)["enabled"]
        # Respect foreign GPU work and the LN reservation as part of the node
        # total. Existing jobs are never killed because someone else starts.
        foreign_gpu = len(observed_gpu - set(live))
        if machine == "local":
            limit = min(limit, max(0, (read(probe)["selected_k"] if probe.exists() else 0) - foreign_gpu))
        missing = sum(j[2]["gpu"] and pid not in observed_gpu for pid, j in live.items())
        active_ids = {j[2]["id"] for j in live.values()}
        for job in queue:
            if failures or (OUT / "PAUSE_ADMISSION").exists():
                break
            if not assigned(job, machine) or job["id"] in active_ids or Path(job["done"]).exists():
                continue
            if job["gpu"]:
                if gpu_count >= limit or not no_swap or not can_start(snapshot, missing):
                    continue
                # A twin already running is allowed to finish before materialization.
                if sharing_enabled and any(j[2].get("fl_fingerprint") == job.get("fl_fingerprint") and
                       j[2].get("cell") == job.get("cell") and j[2].get("seed") == job.get("seed") for j in live.values()):
                    continue
            elif cpu_count >= 2:
                continue
            command = list(job["command"])
            command[0] = sys.executable
            if job["gpu"]:
                shared = sharing_source(job, queue, machine)
                if shared:
                    command += ["--shared-from", shared]
            folder = Path(job["out"])
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / "job.log").open("a")
            proc = subprocess.Popen(command, cwd=WORK, env=environment(), stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            tok = token(proc.pid)
            atomic_json(folder / "process.json", {"pid": proc.pid, "start_token": tok,
                        "command": command, "machine": machine, "queue_job": job["id"]})
            live[proc.pid] = (proc, log, job, tok)
            if job["gpu"]:
                gpu_count += 1
                missing += 1
            else:
                cpu_count += 1
        active = []
        for pid, (_, _, job, _) in live.items():
            progress = Path(job["out"]) / "progress.json"
            active.append({"pid": pid, "id": job["id"], "gpu": job["gpu"],
                           "progress": safe_progress(read(progress)) if progress.exists() else {}})
        atomic_json(status, {"machine": machine, "pid": __import__("os").getpid(), "updated_unix": time.time(),
                    "status": "failed" if failures else "running", "active": active, "failures": failures,
                    "fl_limit": limit, "ln_reserved_slots": 2 if machine == "local" else 0,
                    "resources": snapshot, "no_swapping": no_swap, "memory_guard_ok": snapshot["free_mib"] >= RESERVE_MIB})
        if not live and (failures or (OUT / "ALL_DONE").exists()):
            lock.close()
            return
        time.sleep(3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--machine", choices=("101", "local"), required=True)
    run(p.parse_args().machine)
