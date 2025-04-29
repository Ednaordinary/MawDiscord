from queue import Queue
import threading
import traceback
import asyncio
import random
import time
import sys
import os

from exllamav3 import ComboSampler

from .views import ScrollRedoView
from .history import Message

verbose = True

def run_handler(idx, engine, history, view, token_count):
    if verbose: print("Running handler", idx)
    try:
        temp = random.randint(500, 700) / 1000
        sampler = ComboSampler(temperature=temp, min_p=0.1, top_k=30, top_p=0.95, rep_p=1.0)
        answer = ""
        for i in engine.generate(history, add_bos=False, stop_token="<|im_end|>", max_tokens=4096, sampler=sampler):
            token_count.inc()
            answer += i
            view.update_answer(idx, answer)
        view.update_answer(idx, answer, limit=False)
    except Exception as e:
        print("Error in handler")
        print(traceback.format_exc())
    view.complete_answer(idx)
    token_count.req_count.inc()
    if verbose: print("Handler", idx, "exit")

async def get_scroll_view(context, tools, edit, queue, loop, cutoff, continue_request, req_count):
    # answers, history, user_message, message, prompt, tools, idx, edit, queue, loop, timeout, cutoff, continue_request, req_count
    view_kwargs = {"answers": [""]*5, "context": context, "tools": tools, "edit": edit, "queue": queue, "loop": loop, "timeout": None, "cutoff": cutoff, "continue_request": continue_request, "req_count": req_count}
    return ScrollRedoView(**view_kwargs)

class Request:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def handle(self):
        pass

class RequestContext:
    def __init__(self, message, bot_message, history, prompt):
        self.message = message
        self.bot_message = bot_message
        self.history = history
        self.prompt = prompt

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
        self.context.history.workers.append(int(self.context.bot_message.id))
        self.context.history.add_message(Message(self.context.message.id, self.context.prompt, "user"))
        tool_prompt = self.context.history.sys + "\n\n" + "\n".join([x.doc for x in self.tools if hasattr(x, "doc")])
        self.context.history.edit_message(Message(0, tool_prompt, "system"))
        view = asyncio.run_coroutine_threadsafe(coro=get_scroll_view(self.context, self.tools, self.edit, channel_queue, discord_loop, self.cutoff, ScrollRequest, self.req_count), loop=discord_loop).result()
        history = self.context.history.to_tokenizer(limit=self.context.message.id)
        history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
        threads = []
        token_count = TokenCount(self.req_count)
        for i in range(5):
            if verbose: print("Starting handler:", i)
            threads.append(threading.Thread(target=run_handler, args=[i, engine, history, view, token_count]))
        for i in threads:
            i.start()
        for i in threads:
            i.join()
        self.context.history.workers.remove(int(self.context.bot_message.id))
        return token_count.get()
    def update_progress(self, content, discord_loop):
        asyncio.run_coroutine_threadsafe(coro=self.context.bot_message.edit(content=content), loop=discord_loop)

class ScrollRequest(Request):
    # __init__: # context, view
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        try:
            self.context.history.workers.append(int(self.context.bot_message.id))
            self.context.history.add_message(Message(self.context.message.id, self.prompt, "user"))
            history = self.context.history.to_tokenizer(limit=self.context.message.id)
            history = tokenizer.history_to_tokens(history, cutoff=self.view.cutoff)
            threads = []
            token_count = TokenCount(self.view.req_count)
            for i in self.view.idxs:
                if verbose: print("Starting handler", i)
                threads.append(threading.Thread(target=run_handler, args=[i, engine, history, self.view, token_count]))
            for i in threads:
                i.start()
            for i in threads:
                i.join()
            self.context.history.workers.remove(int(self.context.bot_message.id))
            return token_count.get()
        except:
            print(traceback.format_exc())
    def update_progress(self, content, discord_loop):
        if self.view.get_idx() in self.view.idxs:
            asyncio.run_coroutine_threadsafe(coro=self.context.bot_message.edit(content=content), loop=discord_loop)
