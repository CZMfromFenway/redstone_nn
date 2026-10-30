"""
Prove that time is READ from the system clock, not guessed.

It (a) prints the clock, (b) measures a real elapsed interval and compares it against
the same interval read from the clock, and (c) computes the deadline arithmetic. If the
clock were being faked or hardcoded, the elapsed-time check below would fail.
"""
import time, datetime

print("=" * 66)
print("1. the clock, read three independent ways")
print("=" * 66)
print("  time.time()                  :", time.time(), " (seconds since the Unix epoch)")
print("  time.strftime('%Y-%m-%d %H:%M:%S') :", time.strftime("%Y-%m-%d %H:%M:%S"))
print("  datetime.datetime.now()      :", datetime.datetime.now().isoformat(sep=" ", timespec="seconds"))
lt = time.localtime()
print(f"  broken down from the clock   : year={lt.tm_year} mon={lt.tm_mon} day={lt.tm_mday} "
      f"hour={lt.tm_hour} min={lt.tm_min} sec={lt.tm_sec} wday={lt.tm_wday}")
print(f"  seconds since local midnight : {lt.tm_hour*3600 + lt.tm_min*60 + lt.tm_sec}")

print()
print("=" * 66)
print("2. do the clock and the stopwatch agree? (this is the real test)")
print("=" * 66)
c0 = datetime.datetime.now()
print("  clock before sleep :", c0.strftime("%H:%M:%S.%f")[:-3])
t0 = time.perf_counter()
time.sleep(3.0)                      # a REAL 3-second wait
t1 = time.perf_counter()
c1 = datetime.datetime.now()
print("  clock after sleep  :", c1.strftime("%H:%M:%S.%f")[:-3])
clock_delta = (c1 - c0).total_seconds()
stopwatch_delta = t1 - t0
print(f"  measured by clock     : {clock_delta:.3f} s")
print(f"  measured by stopwatch : {stopwatch_delta:.3f} s")
print(f"  difference            : {abs(clock_delta - stopwatch_delta)*1000:.1f} ms")
ok = abs(clock_delta - 3.0) < 0.5 and abs(stopwatch_delta - 3.0) < 0.5
print(f"  -> both agree with a real 3 s wait: {ok}")
assert ok, "the clock is not tracking real time!"

print()
print("=" * 66)
print("3. deadline arithmetic, from the clock")
print("=" * 66)
DEADLINE = "21:00"
h, m = (int(x) for x in DEADLINE.split(":"))
now = datetime.datetime.now()
deadline = now.replace(hour=h, minute=m, second=0, microsecond=0)
if deadline <= now:
    deadline += datetime.timedelta(days=1)
left = (deadline - now).total_seconds()
print(f"  now           : {now.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  deadline      : {deadline.strftime('%Y-%m-%d %H:%M:%S')}   (configured {DEADLINE})")
print(f"  time left     : {left:.0f} s = {left/60:.1f} min")
print(f"  reserve 180 s -> usable {max(0, left-180):.0f} s = {max(0,left-180)/60:.1f} min")

print()
print("=" * 66)
print("4. what the runner would do with this reading")
print("=" * 66)
print("  IMPORTANT: no s/step value is assumed here. The real runner")
print("  (measure_then_plan.py) first PILOT-RUNS every configuration, derives s/step from")
print("  the measured wall time, and only then plans. Until a pilot has run, a")
print("  configuration's cost is reported as NOT MEASURED rather than guessed.")
for label, per_step, steps in (("sweep_pca8", None, 1500),
                               ("sweep_half", None, 1500),
                               ("kernel_style", None, 4000)):
    if per_step is None:
        print(f"  ???? {label:14s} s/step NOT MEASURED YET -> must be piloted before planning")
        continue
    cost = per_step * steps
    verdict = "RUN " if cost <= max(0, left - 180) else "SKIP"
    print(f"  {verdict} {label:14s} {per_step:.4f} s/step x {steps} = {cost:6.1f} s "
          f"= {cost/60:4.1f} min   (fits: {cost <= max(0, left-180)})")
print()
print("  The real pipeline is:  measure_then_plan.py  ->  reads the clock, pilot-runs each")
print("  configuration for real, writes timings.json, and only then decides.")
print()
print("CLOCK_PROOF_DONE")
