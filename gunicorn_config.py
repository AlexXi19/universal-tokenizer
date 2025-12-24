import os
import signal
import threading
import time

from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics

# Gunicorn config variables
workers = 1
bind = "0.0.0.0:8080"
preload_app = True


def when_ready(server):
    remain_workers = int(os.environ.get('WORKERS', '4')) - workers
    master_pid = os.getpid()

    def create_worker_gradually():
        for _ in range(remain_workers):
            time.sleep(5)
            os.kill(master_pid, signal.SIGTTIN)

    threading.Thread(
        target=create_worker_gradually,
        daemon=True,
    ).start()


# For prometheus metrics
def child_exit(server, worker):
    GunicornInternalPrometheusMetrics.mark_process_dead_on_child_exit(worker.pid)
