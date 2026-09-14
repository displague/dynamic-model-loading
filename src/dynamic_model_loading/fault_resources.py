"""Fail-closed device/host sampling for the Windows fault-pager experiment."""
import ctypes
import json
import os
from pathlib import Path
import threading
import time

import psutil


class Resources:
    class Memory(ctypes.Structure):
        _fields_ = [(k, ctypes.c_ulonglong) for k in ('total', 'free', 'used')]

    def __init__(self, path, gpu_limit=15000 * 2**20, host_floor=2048 * 2**20):
        self.path = Path(path)
        self.gpu_limit, self.host_floor = gpu_limit, host_floor
        self.dll = ctypes.CDLL(str(Path(os.environ['SystemRoot']) / 'System32/nvml.dll'))
        self.check_code(self.dll.nvmlInit_v2())
        self.handle = ctypes.c_void_p()
        self.check_code(self.dll.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)))
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.error = None
        self.peak_gpu = 0
        self.min_host = float('inf')
        self.peak_rss = 0
        self.samples = 0
        self.lock = threading.Lock()

    @staticmethod
    def check_code(code):
        if code:
            raise RuntimeError(f'NVML error {code}')

    def sample(self):
        with self.lock:
            mem = self.Memory()
            self.check_code(self.dll.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)))
            host = psutil.virtual_memory()
            process = self.process.memory_info()._asdict()
            row = dict(monotonic=time.perf_counter(), gpu_used=mem.used,
                       host_available=host.available, process=process)
            self.out.write(json.dumps(row) + '\n')
            self.out.flush()
            self.peak_gpu = max(self.peak_gpu, mem.used)
            self.min_host = min(self.min_host, host.available)
            self.peak_rss = max(self.peak_rss, process['rss'])
            self.samples += 1
            if mem.used > self.gpu_limit or host.available < self.host_floor:
                self.error = 'joint memory allowance exceeded'
            return row

    def loop(self):
        while not self.stop.wait(.2):
            try:
                self.sample()
            except BaseException as exc:
                self.error = repr(exc)
                break

    def check(self):
        """Cheap latched-error check; never add acceptance-dependent sampling work."""
        if self.error:
            raise RuntimeError(self.error)

    def boundary(self):
        self.sample()
        self.check()

    def receipt(self):
        return dict(samples=self.samples, peak_gpu_used=self.peak_gpu,
                    min_host_available=self.min_host, peak_rss=self.peak_rss,
                    passed=bool(self.samples and self.error is None), error=self.error)

    def __enter__(self):
        self.out = self.path.open('x', encoding='utf-8')
        try:
            self.boundary()
        except BaseException:
            self.out.close()
            self.dll.nvmlShutdown()
            raise
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        try:
            self.boundary()
        finally:
            self.out.close()
            self.dll.nvmlShutdown()
