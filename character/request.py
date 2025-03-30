from exllamav2.generator import ExLlamaV2Sampler
from .views import ScrollRedoView
from .history import Message
from queue import Queue
import threading
import asyncio
import random
import time

verbose = False

def run_handler(idx, engine, history, view, token_count):
    if verbose: print("Running handler", idx)
    try:
        temp = random.randint(500, 700) / 1000
        sampler = ExLlamaV2Sampler.Settings(temperature=temp, min_p=0.02, xtc_threshold=0.05, xtc_probability=0.5, top_k=50, top_p=0.95, token_repetition_penalty=1.0, dry_allowed_length=2, dry_multiplier=0.8, dry_base=1.75, dry_sequence_breakers=["\n", ":", "\"", "*"])
        answer = ""
        for i in engine.generate(history, add_bos=False, stop_token="<｜end▁of▁sentence｜>", max_tokens=1024, sampler=sampler):
            token_count.inc()
            answer += i
            view.update_answer(idx, answer)
        view.update_answer(idx, answer, limit=False)
    except Exception as e:
        print("Error in handler")
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
    view.complete_answer(idx)
    if verbose: print("Handler", idx, "exit")

async def get_scroll_view(history, user_message, bot_message, prompt, tools, edit, queue, loop, cutoff, continue_request):
    return ScrollRedoView([""]*5, history, user_message, bot_message, prompt, tools, edit=edit, queue=queue, loop=loop, cutoff=cutoff, continue_request=continue_request)

class Request:
    def __init__(self):
        pass
    def handle(self):
        pass

class TokenCount():
    def __init__(self):
        self.tokens = 0
    def inc(self):
        self.tokens += 1
    def get(self):
        return self.tokens

class CharacterRequest(Request):
    def __init__(self, message, bot_message, history, prompt, cutoff, tools, edit):
        self.message = message
        self.bot_message = bot_message
        self.channel = message.channel
        self.history = history
        self.prompt = prompt
        self.cutoff = cutoff
        self.tools = tools
        self.edit = edit
    def is_server(self):
        return True if self.message.guild != None else None
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        self.history.workers.append(int(self.bot_message.id))
        self.history.add_message(Message(self.message.id, self.prompt, "user"))
        tool_prompt = self.history.sys + "\n\n" + "\n".join([x.doc for x in self.tools if hasattr(x, "doc")])
        self.history.edit_message(Message(0, tool_prompt, "system"))
        view = asyncio.run_coroutine_threadsafe(coro=get_scroll_view(self.history, self.message, self.bot_message, self.prompt, self.tools, self.edit, channel_queue, discord_loop, self.cutoff, ScrollRequest), loop=discord_loop).result()
        history = self.history.to_tokenizer(limit=self.message.id)
        history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
        threads = []
        token_count = TokenCount()
        for i in range(5):
            if verbose: print("Starting handler:", i)
            threads.append(threading.Thread(target=run_handler, args=[i, engine, history, view, token_count]))
        for i in threads:
            i.start()
        for i in threads:
            i.join()
        return token_count.get()
        self.history.workers.remove(int(self.bot_message.id))
    def update_progress(self, content, discord_loop):
        asyncio.run_coroutine_threadsafe(coro=self.bot_message.edit(content=content), loop=discord_loop)

class ScrollRequest(CharacterRequest):
    def __init__(self, message, bot_message, prompt, cutoff, tools, edit, view, idxs):
        super().__init__(message, bot_message, view.history, prompt, cutoff, tools, edit)
        self.view = view
        self.idxs = idxs
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        self.history.workers.append(int(self.bot_message.id))
        self.history.add_message(Message(self.message.id, self.prompt, "user"))
        history = self.view.history.to_tokenizer(limit=self.message.id)
        history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
        threads = []
        token_count = TokenCount()
        for i in self.idxs:
            if verbose: print("Starting handler", i)
            threads.append(threading.Thread(target=run_handler, args=[i, engine, history, self.view, token_count]))
        for i in threads:
            i.start()
        for i in threads:
            i.join()
        return token_count.get()
        self.history.workers.remove(int(self.bot_message.id))
    def update_progress(self, content, discord_loop):
        if self.view.get_idx() in self.idxs:
            asyncio.run_coroutine_threadsafe(coro=self.bot_message.edit(content=content), loop=discord_loop)
