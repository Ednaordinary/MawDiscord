from exllamav3 import Model, Config, Cache, CacheLayer_quant, Tokenizer, AsyncGenerator, AsyncJob
from multiprocessing import Queue
import threading
import traceback
import asyncio
import random
import ctypes
import torch
import sys
import gc
import os

libc = ctypes.CDLL("libc.so.6")

def run_loop(loop):
    loop.run_forever()

class Exl3Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run_loop, args=[self.loop], daemon=True)
        self.thread.start()

class Exl3Engine:
    def __init__(self, model_id, cache_size, quant, loop, callback):
        self.config = Config.from_directory(model_id)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=cache_size, layer_type=CacheLayer_quant, k_bits=quant[0], v_bits=quant[1])
        self.model.load(progressbar=False, callback=callback)
        self.tokenizer = Tokenizer.from_config(self.config)
        self.engineloop = loop
        asyncio.run_coroutine_threadsafe(self._get_gen(), loop=self.engineloop.loop).result()

    async def _get_gen(self):
        self.generator = AsyncGenerator(
            model = self.model,
            cache = self.cache,
            tokenizer = self.tokenizer,
        )

    async def _run_job(self, prompt, stop_token: str, add_bos: bool, max_tokens: int, sampler, queue: Queue):
        try:
            job = AsyncJob(
                self.generator,
                max_new_tokens = max_tokens,
                token_healing=True,
                gen_settings=sampler,
                decode_special_tokens=True,
                input_ids = self.tokenizer.encode(prompt, add_bos = False) if isinstance(prompt, str) else prompt,
                seed=random.randint(1, 10000000),
            )
            async for result in job:
                text = result.get("text", "")
                if stop_token in text:
                    await job.cancel()
                    break
                if text != None and text != "":
                    queue.put(text)
        except Exception as e:
            print("Exception in engine:")
            print(traceback.format_exc())
        queue.put(None)
    
    def generate(self, prompt, stop_token, sampler, add_bos=True, max_tokens=256):
        queue = Queue()
        asyncio.run_coroutine_threadsafe(self._run_job(prompt, stop_token, add_bos, max_tokens, sampler, queue), self.engineloop.loop)
        while True:
            i = queue.get()
            if i == None:
                break
            yield i
    
    def flat_generate(self, prompt):
        text = ""
        for i in self.generate(prompt):
            text = text + i
        return text
    
    async def _exit(self):
        await self.generator.close()
    
    def end(self):
        asyncio.run_coroutine_threadsafe(self._exit(), loop=self.engineloop.loop).result()
        del self.generator
        del self.model
        del self.cache
        del self.config
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        libc.malloc_trim(0)
