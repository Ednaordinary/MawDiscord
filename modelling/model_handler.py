# Currently unused! Meant for a future revision of Maw

import vram
import threading
import time
from exl2model import Exl2Engine
from multiprocessing import Queue

class ModelHandler:
    def __init__(self):
        self.model = None
        self.allocation_lock = False
        self.users = 0
        self.progress = None
    def allocate(progress=False):
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
            return self.model
        elif self.model != None:
            return self.model
        else:
            last_prog = None
            while self.allocation_lock == True:
                time.sleep(0.02)
                if not self.prog == last_prog:
                    last_prog = self.prog
                    yield self.prog
            return self.model
    def deallocate():
        self.users -= 1
        if self.users < 0:
            self.users = 0
        if self.users == 0 and self.model != None and self.allocation_lock == False:
            self.allocation_lock = True
            threading.Thread(target=self.unload).start()

class Exl2ModelHandler(ModelHandler):
    def __init__(self, model_id, cache_size, cache_impl, loop, chunk_size=512):
        super().__init__()
        self.model_id = model_id
        self.cache_size = cache_size
        self.cache_impl = cache_impl
        self.loop = loop
        self.chunk_size = chunk_size
        self.progress = Queue()
    def load_callback(self, current, total):
        self.progress.put(tuple((current, total)))
    def _load(self):
        try:
            self.model = Exl2Engine(self.model_id, self.cache_size, self.cache_impl, self.loop, self.load_callback)
        except:
            pass
        self.progress.put(True)
    def load(self):
        threading.Thread(target=_load).start()
        while True:
            prog = self.progress.get()
            if prog == True:
                break
            else:
                yield prog
    def unload(self):
        self.model.end()
        self.model = None
        self.allocation_lock = False
        vram.deallocate("Maw")

class Exl2ModelHandlerLazy(Exl2ModelHandler):
    def __init__(self, model_id, cache_size, cache_impl, loop, chunk_size=512, timeout=10 * 60):
        super().__init__()
        self.timeout = timeout
        self.timeout_lock = False
        self.current_timeout = None
    def deallocate():
        self.users -= 1
        if self.users < 0:
            self.users = 0
        if self.users == 0 and self.model != None and self.allocation_lock == False:
            threading.Thread(target=self.unload).start()
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
            self.current_timeout = None
            self.allocation_lock = True
            self.model.end()
            self.model = None
            self.allocation_lock = False
        else:
            self.current_timeout = self.timeout + time.perf_counter()
