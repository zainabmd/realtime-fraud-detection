"""
One-command local stack with a tiny process supervisor.

    python run_local.py                       # train if needed, then producer + consumer + API + dashboard
    python run_local.py --fresh --hours 6     # wipe broker/lake/state first, backfill 6 h of history
    python run_local.py --no-dashboard        # headless (e.g. run the dashboard elsewhere)
    python run_local.py --fresh --drift-after-hours 4   # inject drift 4 h into the backfill (monitoring demo)

Each service runs as its own OS process – exactly like separate containers –
communicating only through the broker and the lake. Crashed services are
restarted with exponential backoff; Ctrl+C stops everything gracefully.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import API_PORT, BROKER_DB, DATA_DIR, MODEL_DIR, get_logger

log = get_logger("supervisor")
ROOT = Path(__file__).resolve().parent
PY = sys.executable


@dataclass
class Service:
    name: str
    cmd: list[str]
    restart_cmd: list[str] | None = None   # e.g. producer: never re-run the backfill after a crash
    proc: subprocess.Popen | None = None
    restarts: int = 0
    next_start: float = 0.0
    log_path: Path = field(default=None)

    def start(self) -> None:
        self.log_path = DATA_DIR / "logs" / f"{self.name}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log_path, "a", encoding="utf-8")
        cmd = self.restart_cmd if (self.restarts and self.restart_cmd) else self.cmd
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        log.info("▶ %-10s pid=%-6s log=%s", self.name, self.proc.pid, self.log_path.relative_to(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full streaming stack locally")
    ap.add_argument("--fresh", action="store_true", help="delete broker, lake and feature state first")
    ap.add_argument("--hours", type=float, default=6.0, help="hours of history to backfill before going live")
    ap.add_argument("--tps", type=float, default=25.0, help="live transactions per second")
    ap.add_argument("--retrain", action="store_true", help="retrain the model at --tps before starting")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--dashboard-port", type=int, default=8501)
    ap.add_argument("--drift-after-hours", type=float, default=None, help="simulate drift this many hours into the run")
    args = ap.parse_args()

    if args.fresh:
        for p in (DATA_DIR / "lake", DATA_DIR / "state"):
            shutil.rmtree(p, ignore_errors=True)
        for p in BROKER_DB.parent.glob(BROKER_DB.name + "*"):
            p.unlink(missing_ok=True)
        log.info("Fresh start: broker, lake and state wiped")
        import importlib

        import src.config
        importlib.reload(src.config)  # recreate directories

    if args.retrain or not (MODEL_DIR / "fraud_model.joblib").exists() or not (MODEL_DIR / "challenger.joblib").exists():
        # Train at the serving traffic rate: velocity features (txns/min, secs since last) scale with TPS,
        # so a model trained on quieter traffic sees normal busy-hour behaviour as anomalous.
        log.info("Training the model at %.0f TPS (≈2-3 min)", args.tps)
        if subprocess.call([PY, "-m", "src.train_model", "--tps", str(args.tps), "--hours", "6"], cwd=ROOT) != 0:
            log.error("Training failed")
            return 1

    live_cmd = [PY, "-m", "src.simulator", "live", "--tps", str(args.tps)]
    has_history = any((DATA_DIR / "lake" / "scored").glob("dt=*/hr=*/*.parquet"))
    if has_history:
        log.info("Existing lake found – resuming live without a new backfill (use --fresh to rebuild)")
    producer_cmd = live_cmd if has_history else [PY, "-m", "src.simulator", "run", "--hours", str(args.hours),
                                                 "--tps", str(args.tps)]
    if args.drift_after_hours is not None:
        producer_cmd = producer_cmd + ["--drift-after-hours", str(args.drift_after_hours)]
    services = [
        Service("consumer", [PY, "-m", "src.consumer"]),
        Service("producer", producer_cmd, restart_cmd=live_cmd),
        Service("graph", [PY, "-m", "src.graph", "--watch", "120"]),          # fraud-ring snapshots
        Service("monitor", [PY, "-m", "src.monitoring", "--watch", "300"]),   # drift + champion/challenger
    ]
    if not args.no_api:
        services.append(Service("api", [PY, "-m", "uvicorn", "src.api:app", "--port", str(API_PORT), "--log-level", "warning"]))
    if not args.no_dashboard:
        services.append(Service("dashboard", [PY, "-m", "streamlit", "run", "dashboard/app.py",
                                              "--server.port", str(args.dashboard_port), "--server.headless", "true"]))

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    for s in services:
        s.start()
        time.sleep(1.0)  # consumer subscribes before the producer floods the topic
    if not args.no_dashboard:
        log.info("Dashboard → http://localhost:%d   API docs → http://localhost:%d/docs", args.dashboard_port, API_PORT)

    while not stopping:
        for s in services:
            rc = s.proc.poll() if s.proc else None
            if rc is not None and time.time() >= s.next_start:
                if s.name == "producer" and rc == 0:
                    continue  # finite runs (--duration) end normally
                s.restarts += 1
                backoff = min(60, 2 ** s.restarts)
                log.warning("✖ %s exited (rc=%s) – restart #%d in %ds (see %s)", s.name, rc, s.restarts, backoff, s.log_path.name)
                s.next_start = time.time() + backoff
                s.proc = None
            elif s.proc is None and time.time() >= s.next_start:
                s.start()
        time.sleep(1)

    log.info("Stopping services…")
    for s in reversed(services):  # dashboard/API first, consumer last so it can drain
        if s.proc and s.proc.poll() is None:
            s.proc.terminate()  # SIGTERM on POSIX (graceful drain); TerminateProcess on Windows (replay-safe)
    deadline = time.time() + 20
    for s in services:
        if s.proc:
            try:
                s.proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                s.proc.kill()
    log.info("All services stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
