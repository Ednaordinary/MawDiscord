import threading
import time

from ..vram import Vram

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
                if not self.progress == last_prog:
                    last_prog = self.progress
                    yield self.progress
            yield self.model
    def deallocate(self):
        self.users -= 1
        if self.users < 0:
            self.users = 0
        if self.users == 0 and self.model != None and self.allocation_lock == False:
            threading.Thread(target=self.unload).start()
    def _vram_unload(self):
        vram.deallocate("Maw")
    def _vram_get_alloc(self):
        return vram.get_allocations()
