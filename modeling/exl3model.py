from multiprocessing import Queue, Process
import threading
import traceback
import asyncio
import random
import ctypes
import torch
import time
import sys
import gc
import os

from exllamav3 import Model, Config, Cache, CacheLayer_quant, Tokenizer, AsyncGenerator, AsyncJob, ComboSampler

libc = ctypes.CDLL("libc.so.6")

resp_count = 256 # hard limit. will decrease generations even if higher in request and views. default 256

def run_loop(self):
    try:
        self.loop.run_forever()
    except:
        print(traceback.format_exc(), flush=True)

class Exl3Loop:
    def __init__(self):
        try:
            self.loop = asyncio.new_event_loop()
            #self.process = Process(target=run_loop, args=[self])
            #self.process.start()
            self.thread = threading.Thread(target=run_loop, args=[self], daemon=True)
            self.thread.start()
        except:
            print(traceback.format_exc(), flush=True)

class Exl3Engine:
    def __init__(self, model_id, cache_size, quant, loop, callback):
        self.config = Config.from_directory(model_id)
        self.model = Model.from_config(self.config)
        if quant:
            self.cache = Cache(self.model, max_num_tokens=cache_size, layer_type=CacheLayer_quant, k_bits=quant[0], v_bits=quant[1])
        else:
            self.cache = Cache(self.model, max_num_tokens=cache_size)
        self.model.load(progressbar=False, callback=callback)
        self.tokenizer = Tokenizer.from_config(self.config)
        self.engineloop = loop
        asyncio.run_coroutine_threadsafe(self._get_gen(), loop=self.engineloop.loop).result()

    async def _get_gen(self):
        try:
            self.generator = AsyncGenerator(
                model = self.model,
                cache = self.cache,
                tokenizer = self.tokenizer,
                max_batch_size = resp_count,
            )
        except:
            print(traceback.format_exc(), flush=True)

    async def _run_job(self, prompt, stop_token: str, add_bos: bool, max_tokens: int, sampler, queue: Queue, looped=False):
        ids = self.tokenizer.encode(prompt, add_bos = add_bos).to("cpu") if isinstance(prompt, str) else prompt.to("cpu") if isinstance(prompt, torch.Tensor) else prompt["input_ids"]
        try:
            job = AsyncJob(
                self.generator,
                max_new_tokens = max_tokens,
                token_healing=True,
                sampler=sampler,
                decode_special_tokens=True,
                input_ids = ids,
                seed=random.randint(1, 10000000),
            )
            think_switch = looped
            pre_think = 0
            pre_think_text = ""
            post_think = 0
            async for result in job:
                text = result.get("text", "")
                if think_switch:
                    post_think += len(text)
                else:
                    pre_think += len(text)
                    print(pre_think)
                    pre_think_text += text
                if text == "</think>":
                    think_switch = True
                if post_think >= 2000: # Force stop after 2000 sent characters
                    await job.cancel()
                    queue.put(True)
                    return
                if not think_switch and pre_think >= 8000:
                    await job.cancel()
                    queue.put("</think>")
                    new_ids = self.tokenizer.encode(pre_think_text + ". It seems I've been cut off. It's time to respond, even if I'm not quite done. I'll do my best to still get the right answer</think>", add_bos=False)
                    new_ids = new_ids if isinstance(new_ids, torch.Tensor) else new_ids["input_ids"]
                    print(ids.shape, new_ids.shape)
                    new_ids = torch.cat([ids, new_ids], dim=1)
                    return await self._run_job(new_ids, stop_token, False, max_tokens, sampler, queue, looped=True)
                if stop_token in text:
                    await job.cancel()
                    queue.put(True)
                    return
                if text != None and text != "":
                    queue.put(text)
        except Exception as e:
            print("Exception in engine:")
            print(traceback.format_exc())
        queue.put(False)
        
    async def _kickstart(self, prompt):
        job = AsyncJob(
            self.generator,
            max_new_tokens = 100,
            input_ids = self.tokenizer.encode(prompt, add_bos = True).to("cpu") if isinstance(prompt, str) else prompt.to("cpu"),
            seed=random.randint(1, 10000000),
        )
        count = 0
        async for result in job:
            count += 1
            if count >= 2:
                await job.cancel() # a few tokens have been generated, so by this point the pages should be shared
        
    def kickstart(self, prompt):
        asyncio.run_coroutine_threadsafe(self._kickstart(prompt), self.engineloop.loop)
    
    def generate(self, prompt, stop_token, sampler, add_bos=True, max_tokens=256):
        queue = Queue()
        asyncio.run_coroutine_threadsafe(self._run_job(prompt, stop_token, add_bos, max_tokens, sampler, queue), self.engineloop.loop)
        while True:
            i = queue.get()
            if isinstance(i, bool):
                #yield i
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
        self.model.unload()
        del self.generator
        del self.model
        del self.cache
        del self.config
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        libc.malloc_trim(0)
