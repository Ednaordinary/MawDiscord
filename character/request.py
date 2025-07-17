from queue import Queue
import threading
import traceback
import asyncio
import random
import time
import sys
import os

from exllamav3 import ComboSampler
#from exllamav2.generator import ExLlamaV2Sampler
from exllamav3.generator.sampler.custom import *

from .views import ScrollRedoView
from .history import Message

verbose = True
resp_count = 3
#stop_token = "<｜end▁of▁sentence｜>"
stop_token = "<|im_end|>"

class SillySampler(ComboSampler):
    """
    Single class with an argument for each sampling step
    """
    def __init__(
        self,
        rep_p: float = 1.0,
        freq_p: float = 0.0,
        pres_p: float = 0.0, # be careful with this!
        rep_sustain_range: int = 0,
        rep_decay_range: int = 0,
        temperature: float = 0.6,
        min_p: float = 0.01,
        top_k: int = 40,
        top_p: float = 0.95,
    ):
        # Steps with default parameters become no-ops
        stack = []

        if temperature == 0.0:
            stack += [
                SS_Argmax()
            ]
        else:
            stack += [
                SS_TopK(top_k),
                SS_TopP(top_p),
                SS_MinP(min_p),
                SS_Temperature(temperature),
                SS_RepP(rep_p, rep_sustain_range, rep_decay_range),
                SS_PresFreqP(pres_p, freq_p, rep_sustain_range, rep_decay_range),
                SS_Sample()
            ]

        super().__init__(stack)

def run_handler(idx, engine, history, view, token_count, tokenizer, char):
    if verbose: print("Running handler", idx)
    try:
        if char:
            temp = random.randint(1300, 2000) / 1000
            #sampler = ComboSampler(temperature=temp, min_p=0.0, top_k=20, top_p=0.95, rep_p=1.01)
            sampler = SillySampler(temperature=temp, min_p=0.1, top_k=100, top_p=0.95, rep_p=1.1)
        else:
            temp = random.randint(500, 800) / 1000
            #sampler = ComboSampler(temperature=temp, min_p=0.0, top_k=20, top_p=0.95, rep_p=1.01)
            sampler = SillySampler(temperature=temp, min_p=0.0, top_k=20, top_p=0.95)
        #sampler = ExLlamaV2Sampler.Settings(temperature=temp, min_p=0.02, top_k=20, top_p=0.95, token_repetition_penalty=1.0, token_presence_penalty= 2.0)
        answer = ""
        count = 0
        for i in engine.generate(history, add_bos=False, stop_token=stop_token, max_tokens=1024 * 32, sampler=sampler):
            if isinstance(i, bool):
                pass
            else:
                count += 1
                token_count.inc()
                answer += i
                view.update_answer(idx, answer, count)
        view.update_answer(idx, answer, count, limit=False)
    except Exception as e:
        print("Error in handler")
        print(traceback.format_exc())
    view.complete_answer(idx)
    token_count.req_count.inc()
    if verbose: print("Handler", idx, "exit")

def fake_handler(engine, history):
    # this is just to kick start the first few cache pages, otherwise the first gen becomes very slow for some reason
    pass

async def get_scroll_view(context, tools, edit, queue, loop, cutoff, continue_request, req_count):
    # answers, history, user_message, message, prompt, tools, idx, edit, queue, loop, timeout, cutoff, continue_request, req_count
    view_kwargs = {"answers": [""]*resp_count, "context": context, "tools": tools, "edit": edit, "queue": queue, "loop": loop, "timeout": None, "cutoff": cutoff, "continue_request": continue_request, "req_count": req_count}
    return ScrollRedoView(**view_kwargs)

class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def handle(self):
        pass

class RequestContext:
    def __init__(self, message, bot_message, history, prompt, char):
        self.message = message
        self.bot_message = bot_message
        self.history = history
        self.prompt = prompt
        self.char = char

class TokenCount():
    def __init__(self, req_count):
        self.tokens = 0
        self.req_count = req_count
    def inc(self):
        self.tokens += 1
    def get(self):
        return self.tokens

class CharacterRequest(Request):
    # __init__: context, cutoff, tools, edit, req_count
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        try:
            self.context.history.workers.append(int(self.context.bot_message.id))
            self.context.history.add_message(Message(self.context.message.id, self.context.prompt, "user"))
            tool_prompt = self.context.history.sys + "\n\n" + "\n".join([x.doc for x in self.tools if hasattr(x, "doc")])
            self.context.history.edit_message(Message(0, tool_prompt, "system"))
            view = asyncio.run_coroutine_threadsafe(coro=get_scroll_view(self.context, self.tools, self.edit, channel_queue, discord_loop, self.cutoff, ScrollRequest, self.req_count), loop=discord_loop).result()
            history = self.context.history.to_tokenizer(limit=self.context.message.id)
            history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
            threads = []
            token_count = TokenCount(self.req_count)
            engine.kickstart(history) # increase first response speed (sillies)
            for i in range(resp_count):
                if verbose: print("Starting handler:", i)
                threads.append(threading.Thread(target=run_handler, args=[i, engine, history, view, token_count, tokenizer, self.context.char]))
            for i in threads:
                time.sleep(0.01) # Start in order so they also appear completed in order
                i.start()
            for i in threads:
                i.join()
            self.context.history.workers.remove(int(self.context.bot_message.id))
            return token_count.get()
        except Exception as e:
            print(repr(e))
            print(traceback.format_exc())
            return token_count.get()
    def update_progress(self, content, discord_loop):
        asyncio.run_coroutine_threadsafe(coro=self.context.bot_message.edit(content=content), loop=discord_loop)

class ScrollRequest(Request):
    # __init__: # context, view, idxs
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        try:
            self.context.history.workers.append(int(self.context.bot_message.id))
            self.context.history.add_message(Message(self.context.message.id, self.context.prompt, "user"))
            history = self.context.history.to_tokenizer(limit=self.context.message.id)
            history = tokenizer.history_to_tokens(history, cutoff=self.view.cutoff)
            threads = []
            token_count = TokenCount(self.view.req_count)
            engine.kickstart(history) # increase first response speed (sillies)
            for i in self.idxs:
                if verbose: print("Starting handler", i)
                threads.append(threading.Thread(target=run_handler, args=[i, engine, history, self.view, token_count, tokenizer, self.context.char]))
            for i in threads:
                time.sleep(0.01)
                i.start()
            for i in threads:
                i.join()
            self.context.history.workers.remove(int(self.context.bot_message.id))
            return token_count.get()
        except Exception as e:
            print(traceback.format_exc())
    def update_progress(self, content, discord_loop):
        if self.view.get_idx() in self.idxs:
            asyncio.run_coroutine_threadsafe(coro=self.context.bot_message.edit(content=content), loop=discord_loop)
