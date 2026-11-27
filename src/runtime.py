# Spectral-MoE: src/runtime.py
# Author: Zachary Broveak
#
# Session limits: pause at a saved boundary and resume later.
#
# A full configuration takes far longer than a typical cluster job, so every long loop in the
# project is resumable, and this module provides the shared stop signal. configure() (called
# by the pipeline when SESSION_SECONDS is set) stores an absolute deadline in the
# SPECTRAL_MOE_DEADLINE environment variable, so every child process the pipeline launches
# inherits the same deadline. Independently, setting SPECTRAL_MOE_STOP_FILE to a path requests
# a stop whenever that file exists, which stops a run early without killing it.
#
# Loops call due() after finishing an iteration and saving their state, then pause() if it
# returned true; checkpoint() combines the two at points where nothing needs saving. In
# distributed training due(device) all-reduces the flag so every rank stops at the same step.
# A pause writes session-yield.json in the output directory and exits with status 75
# (EX_TEMPFAIL in sysexits.h). The parent pipeline uses child_status() to tell this
# deliberate pause apart from a crash, and Make reports it as status 2. Rerunning the same
# command continues from the saved state.

import json
import os
import time
from pathlib import Path


def configure(out, seconds):
    if seconds < 0:
        raise ValueError("session seconds must be nonnegative")
    if seconds:
        os.environ["SPECTRAL_MOE_DEADLINE"] = str(time.time() + seconds)
        # Clear the previous pause marker before starting a new session.
        marker = Path(out) / "session-yield.json"
        marker.unlink(missing_ok=True)
        os.environ["SPECTRAL_MOE_YIELD_MARKER"] = str(marker)


def due(device=None):
    deadline = float(os.environ.get("SPECTRAL_MOE_DEADLINE", "0"))
    stop_file = os.environ.get("SPECTRAL_MOE_STOP_FILE")
    stop = bool((deadline and time.time() >= deadline)
                or (stop_file and Path(stop_file).exists()))
    if device is not None:
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            # Pause every rank if any rank requests a stop.
            flag = torch.tensor(int(stop), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
    return stop


def pause():
    if int(os.environ.get("RANK", "0")) == 0:
        name = os.environ.get("SPECTRAL_MOE_YIELD_MARKER")
        if name:
            path = Path(name)
            # Write the pause marker completely before making it visible.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"saved_at": time.time()}) + "\n")
            tmp.replace(path)
        print("Session limit reached at a saved boundary; resume the same command.", flush=True)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    raise SystemExit(75)


def checkpoint():
    if due():
        pause()


def child_status(returncode):
    # Distinguish a saved pause from an ordinary child-process failure.
    name = os.environ.get("SPECTRAL_MOE_YIELD_MARKER")
    return 75 if returncode and name and Path(name).exists() else returncode
