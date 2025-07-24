import threading
import traceback
import time
from queue import Queue
from ..exl2model import Exl2Engine
from .base import ModelHandler

class Exl2ModelHandler(ModelHandler):
    def __init__(self, model_id, cache_size, cache_impl, loop, chunk_size=512):
        super().__init__()
        self.model_id = model_id
        self.cache_size = cache_size
        self.cache_impl = cache_impl
        self.loop = loop
        self.chunk_size = chunk_size
        self.load_progress = Queue()
    def load_callback(self, current, total):
        if total != 0:
            print(int(100*(current / total)), end="\r")
            self.load_progress.put(tuple((current, total)))
    def _load(self):
        try:
            self.model = Exl2Engine(self.model_id, self.cache_size, self.cache_impl, self.loop, self.load_callback)
        except Exception as e:
            print(repr(e))
            print(traceback.format_exc())
        self.load_progress.put(True)
        print()
    def load(self):
        threading.Thread(target=self._load).start()
        while True:
            prog = self.load_progress.get()
            if prog == True:
                break
            else:
                yield prog
    def unload(self):
        self.allocation_lock = True
        self.model.end()
        self.model = None
        self.allocation_lock = False
        self._vram_unload()

class Exl2ModelHandlerLazy(Exl2ModelHandler):
    def __init__(self, model_id, cache_size, cache_impl, loop, chunk_size=512, timeout=10 * 60):
        super().__init__(model_id, cache_size, cache_impl, loop, chunk_size)
        self.timeout = timeout
        self.timeout_lock = False
        self.current_timeout = None
    def deallocate(self):
        self.users -= 1
        if self.users < 0:
            self.users = 0
        if self.users == 0 and self.model != None and self.allocation_lock == False:
            threading.Thread(target=self.unload).start()
        if self.alloc_at != None:
            return time.perf_counter() - self.alloc_at
        else:
            return 0
        self.alloc_at = None
    def unload(self):
        if not self.timeout_lock:
            self.timeout_lock = True
            self.current_timeout = self.timeout + time.perf_counter()
            while True:
                time.sleep(0.5)
                if self.current_timeout < time.perf_counter():
                    break
                if len([x for x in self._vram_get_alloc() if x != "Maw"]) != 0:
                    break
            if self.users < 1:
                self.current_timeout = None
                self.allocation_lock = True
                self.model.end()
                self.model = None
                self.allocation_lock = False
                self._vram_unload()
        else:
            self.current_timeout = self.timeout + time.perf_counter()
