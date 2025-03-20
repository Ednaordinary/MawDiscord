import threading
import time
from queue import Queue
from .exl2model import Exl2Engine
from .vram import Vram

vram = Vram()

class ModelHandler:
    def __init__(self):
        self.model = None
        self.allocation_lock = False
        self.users = 0
        self.progress = None
        self.alloc_at = None
    def allocate(self, progress=False):
        if self.users < 0: # well that's not right
            self.users = 1
        else:
            self.users += 1
        if self.model == None and self.allocation_lock == False:
            self.allocation_lock = True
            vram.allocate("Maw")
            for i in vram.wait_for_allocation("Maw"):
                if progress:
                    yield (False, i)
                self.progress = (False, i)
            for i in self.load():
                if progress:
                    yield (True, i)
                self.progress = (True, i)
            self.allocation_lock = False
            if self.alloc_at == None: self.alloc_at = time.perf_counter()
            yield self.model
        elif self.model != None:
            yield self.model
        else:
            last_prog = None
            while self.allocation_lock == True:
                time.sleep(0.02)
                if not self.prog == last_prog:
                    last_prog = self.prog
                    yield self.prog
            yield self.model
    def deallocate(self):
        self.users -= 1
        if self.users < 0:
            self.users = 0
        if self.users == 0 and self.model != None and self.allocation_lock == False:
            threading.Thread(target=self.unload).start()

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
        print(int(100*(current / total)), end="\r")
        self.load_progress.put(tuple((current, total)))
    def _load(self):
        try:
            self.model = Exl2Engine(self.model_id, self.cache_size, self.cache_impl, self.loop, self.load_callback)
        except Exception as e:
            print(repr(e))
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
        vram.deallocate("Maw")

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
                if len([x for x in vram.get_allocations() if x != "Maw"]) != 0:
                    break
            if self.users < 1:
                self.current_timeout = None
                self.allocation_lock = True
                self.model.end()
                self.model = None
                self.allocation_lock = False
                vram.deallocate("Maw")
        else:
            self.current_timeout = self.timeout + time.perf_counter()
