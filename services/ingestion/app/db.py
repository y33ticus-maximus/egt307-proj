"""Small synchronous PostgreSQL connection pool used by the Flask service.

The pool is opened on a background thread rather than at startup. If PostgreSQL
is not accepting connections yet, the service still starts and answers /health,
and holds /ready at 503 until the pool opens.

That matters because Kubernetes gives no ordering guarantee between pods. A
service that exits when its database is missing cannot be restarted into a
working state, so it sits in CrashLoopBackOff on a cluster that is otherwise
healthy. Withholding traffic and waiting is the correct response; dying is not.
"""

import logging
import os
import threading
import time
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import PoolError, ThreadedConnectionPool

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://farm:farm@postgres:5432/farmdb")

_pool = None
_started = False
_lock = threading.Lock()


class NotReady(RuntimeError):
    """Raised when the pool has not opened yet. Endpoints turn this into a 503."""


def _keep_trying(delay=2):
    """Retry forever. PostgreSQL runs init.sql on first boot, which can take
    longer than any fixed number of attempts worth hard-coding."""
    global _pool
    attempt = 0
    while _pool is None:
        attempt += 1
        try:
            pool = ThreadedConnectionPool(1, 10, DATABASE_URL)
            with _lock:
                _pool = pool
            log.info("database connected after %d attempt(s)", attempt)
            return
        except Exception as exc:
            if attempt in (1, 5, 15) or attempt % 30 == 0:
                log.warning("database not reachable yet (attempt %d): %s", attempt, exc)
            time.sleep(delay)


def connect(background=True):
    """Open the pool. background=True returns at once and keeps retrying."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    if background:
        threading.Thread(target=_keep_trying, daemon=True).start()
    else:
        _keep_trying()


def ready():
    """Has the pool opened yet?"""
    return _pool is not None


@contextmanager
def connection():
    """Borrow a connection and always return it safely to the pool."""
    if _pool is None:
        raise NotReady("database unavailable")

    conn = _pool.getconn()
    broken = False
    try:
        yield conn
        conn.commit()
    except Exception:
        broken = True
        try:
            conn.rollback()
        except psycopg2.Error:
            broken = True
        raise
    finally:
        _pool.putconn(conn, close=broken)


def is_healthy():
    try:
        with connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        return True
    except (psycopg2.Error, PoolError, NotReady, RuntimeError):
        return False


def close():
    global _pool, _started
    if _pool is not None:
        _pool.closeall()
        _pool = None
    _started = False
