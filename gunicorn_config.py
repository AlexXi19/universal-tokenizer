import os
from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics

# Gunicorn config variables
workers = int(os.environ.get('WORKERS', '4'))
bind = "0.0.0.0:8080"
preload_app = True

# For prometheus metrics
def child_exit(server, worker):
    GunicornInternalPrometheusMetrics.mark_process_dead_on_child_exit(worker.pid) 