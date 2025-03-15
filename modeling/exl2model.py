from exllamav2.generator import ExLlamaV2DynamicGeneratorAsync, ExLlamaV2DynamicJobAsync
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from multiprocessing import Queue
import threading
import asyncio
import sys
import gc
import torch
import ctypes
import random

libc = ctypes.CDLL("libc.so.6")

def run_loop(loop):
    loop.run_forever()

class Exl2Loop:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run_loop, args=[self.loop], daemon=True)
        self.thread.start()

class Exl2Engine:
    def __init__(self, model_id, cache_size, cache_impl, loop, callback, chunk_size=512, draft_model_id=None):
        self.config = ExLlamaV2Config(model_id)
        self.config.arch_compat_overrides()
        self.config.max_input_len = chunk_size
        self.config.max_attention_size = chunk_size ** 2
        self.model = ExLlamaV2(self.config)
        self.cache = cache_impl(self.model, lazy=True, max_seq_len=cache_size)
        self.model.load_autosplit(self.cache, progress = False, callback=callback)
        if draft_model_id != None:
            self.draft_config = ExLlamaV2Config(draft_model_id)
            self.draft_config.arch_compat_overrides()
            self.draft_config.max_input_len = chunk_size
            self.draft_config.max_attention_size = chunk_size ** 2
            self.draft_model = ExLlamaV2(self.draft_config)
            self.draft_cache = cache_impl(self.draft_model, lazy=True, max_seq_len=cache_size)
            self.draft_model.load_autosplit(self.draft_cache, progress=False)
        else:
            self.draft_model = None
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.engineloop = loop
        asyncio.run_coroutine_threadsafe(self._get_gen(cache_impl, cache_size), loop=self.engineloop.loop).result()

    async def _get_gen(self, cache_impl, cache_size):
        if self.draft_model:
            self.generator = ExLlamaV2DynamicGeneratorAsync(
                model = self.model,
                cache = self.cache,
                draft_model = self.draft_model,
                draft_cache = self.draft_cache,
                tokenizer = self.tokenizer,
                num_draft_tokens = 2,
            )
        else:
            self.generator = ExLlamaV2DynamicGeneratorAsync(
                model = self.model,
                cache = self.cache,
                tokenizer = self.tokenizer,
            )

    async def _run_job(self, prompt, stop_token: str, add_bos: bool, max_tokens: int, sampler, queue: Queue):
        job = ExLlamaV2DynamicJobAsync(
            self.generator,
            max_new_tokens = max_tokens,
            token_healing=True,
            gen_settings=sampler,
            decode_special_tokens=True,
            input_ids = self.tokenizer.encode(prompt, add_bos = False) if isinstance(prompt, str) else prompt,
            seed=random.randint(1, 10000000),
        )
        try:
            async for result in job:
                text = result.get("text", "")
                if stop_token in text:
                    await job.cancel()
                    break
                if text != None and text != "":
                    queue.put(text)
        except:
            pass
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
