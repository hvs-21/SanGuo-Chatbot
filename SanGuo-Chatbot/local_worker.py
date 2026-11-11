"""Exchange JSON with a persistent local Python process over standard streams."""

import json
import os
import subprocess
import sys
from queue import Empty, Queue
from threading import Lock, Thread

from config import PROJECT_ROOT


def uses_wsl(prefix):
    return os.name == "nt" and bool(os.environ.get(f"{prefix}_WSL_DISTRO"))


class LocalWorker:
    def __init__(self, prefix, script, startup_timeout, query_timeout, variables):
        executable = os.environ.get(f"{prefix}_PYTHON", sys.executable)
        command = [executable, "-B", "-u", script]
        if uses_wsl(prefix):
            distro = os.environ[f"{prefix}_WSL_DISTRO"]
            command = ["wsl.exe", "-d", distro, "--cd", str(PROJECT_ROOT), "--", "env",
                       *(f"{name}={value}" for name, value in variables.items()), *command]
        environment = dict(os.environ, **variables)
        environment.pop("DASHSCOPE_API_KEY", None)
        self._query_timeout = query_timeout
        self._messages = Queue()
        self._lock = Lock()
        self._process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self._reader = Thread(target=self._read_output, daemon=True)
        self._reader.start()
        try:
            if self._receive(startup_timeout) != {"ready": True}:
                raise ValueError("本地模型进程未就绪")
        except BaseException:
            self.close()
            raise

    @property
    def available(self):
        return self._process.poll() is None and self._reader.is_alive()

    def _read_output(self):
        try:
            for line in self._process.stdout:
                self._messages.put(line)
        except (OSError, UnicodeError, ValueError):
            pass
        finally:
            self._messages.put(None)

    def _receive(self, timeout):
        try:
            line = self._messages.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("本地模型响应超时") from exc
        if line is None:
            raise OSError("本地模型进程已退出")
        result = json.loads(line)
        if not isinstance(result, dict):
            raise ValueError("本地模型响应格式错误")
        return result

    def request(self, payload):
        if not self._lock.acquire(timeout=self._query_timeout):
            return {}
        try:
            if not self.available:
                return {}
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
                return self._receive(self._query_timeout)
            except (OSError, ValueError):
                self._stop()
                return {}
        finally:
            self._lock.release()

    def _stop(self):
        if not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._reader.join(timeout=1)
        if not self._reader.is_alive():
            self._process.stdout.close()

    def close(self):
        with self._lock:
            self._stop()
