#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
r"""
Overnight experiment infrastructure.

The user is asleep and cannot approve anything, so "run all night" has to mean:
  * every decision is taken from the SYSTEM CLOCK, never from an estimate I made up
  * every experiment's cost is MEASURED (a pilot run) before it is scheduled
  * a durable log records what ran, when, how long, and how long was left
  * if the machine looks unhealthy (very low free disk), STOP rather than push on
  * nothing is ever overwritten; new outputs always get new names
"""
import json, os, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "results", "overnight", "runner_log.jsonl")


def clock():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def now_sec():
    t = time.localtime()
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec


def hhmm_to_sec(s):
    h, m = (int(x) for x in s.split(":"))
    return h * 3600 + m * 60


def sec_to_hhmm(s):
    s = int(s) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def seconds_until(hhmm):
    """Seconds from now until the next occurrence of hhmm (today or tomorrow)."""
    d = hhmm_to_sec(hhmm)
    n = now_sec()
    if d <= n:
        d += 86400
    return d - n


def disk_free_gb(path=None):
    p = path or ROOT
    try:
        return shutil.disk_usage(p).free / 1e9
    except Exception:
        return float("nan")


def ensure_log():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(event, detail="", **kw):
    ensure_log()
    rec = {"clock": clock(), "t": time.time(), "event": event, "detail": detail, **kw}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['clock']}] {event}: {detail}", flush=True)
    return rec


def read_log():
    if not os.path.exists(LOG):
        return []
    out = []
    for line in open(LOG, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def health_ok(min_free_gb=3.0):
    """Refuse to continue if the disk is nearly full: a runaway job could fill it."""
    free = disk_free_gb()
    ok = (free != free) or (free > min_free_gb)      # nan -> do not block
    if not ok:
        log("health", f"LOW DISK: {free:.2f} GB free (< {min_free_gb} GB) -- stopping")
    return ok


def load_timings(path):
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_timings(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    # self-test: prove the clock is read, not assumed
    print(f"clock now        : {clock()}")
    print(f"seconds of day   : {now_sec()}")
    print(f"disk free        : {disk_free_gb():.1f} GB")
    print(f"until 07:00      : {seconds_until('07:00')/3600:.2f} h")
    print(f"until 08:30      : {seconds_until('08:30')/3600:.2f} h")
    log("selftest", "infrastructure clock check")
    print(f"log file         : {LOG}")
    print("RUNTIME_SELFTEST_DONE")
