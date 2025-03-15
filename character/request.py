from exllamav2.generator import ExLlamaV2Sampler
from .history import Message
from .views import ScrollRedoView
import threading
import asyncio
import random
import time

def run_handler(idx, engine, history, view):
    try:
        temp = random.randint(500, 900) / 1000
        sampler = ExLlamaV2Sampler.Settings(temperature=temp, min_p=0.02, xtc_threshold=0.1, xtc_probability=0.5, top_k=50, top_p=0.95, token_repetition_penalty=1.0, dry_allowed_length=2, dry_multiplier=0.8, dry_base=1.75, dry_sequence_breakers=["\n", ":", "\"", "*"])
        answer = ""
        for i in engine.generate(history, add_bos=False, stop_token="<｜end▁of▁sentence｜>", max_tokens=1024, sampler=sampler):
            answer = answer + i
            view.update_answer(idx, answer)
        view.update_answer(idx, answer, limit=False)
        view.complete_answer(idx)
    except Exception as e:
        print(e)
        print(repr(e))
        view.complete_answer(idx)

async def get_scroll_view(history, user_message, bot_message, prompt, tools, edit, queue, loop, cutoff, continue_request):
    return ScrollRedoView([""]*5, history, user_message, bot_message, prompt, tools, edit=edit, queue=queue, loop=loop, cutoff=cutoff, continue_request=continue_request)

class Request:
    def __init__(self):
        pass
    def handle(self):
        pass

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
        self.history.add_message(Message(self.message.id, self.prompt, "user"))
        view = asyncio.run_coroutine_threadsafe(coro=get_scroll_view(self.history, self.message, self.bot_message, self.prompt, self.tools, self.edit, channel_queue, discord_loop, self.cutoff, ScrollRequest), loop=discord_loop).result()
        history = self.history.to_tokenizer(limit=self.message.id)
        history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=run_handler, args=[i, engine, history, view]))
        for i in threads:
            i.start()
            time.sleep(0.02)
        for i in threads:
            i.join()
    def update_progress(self, content, discord_loop):
        asyncio.run_coroutine_threadsafe(coro=self.bot_message.edit(content), loop=discord_loop)

class ScrollRequest(CharacterRequest):
    def __init__(self, message, bot_message, prompt, cutoff, tools, edit, view, idxs):
        super().__init__(message, bot_message, view.history, prompt, cutoff, tools, edit)
        self.view = view
        self.idxs = idxs
    def handle(self, engine, tokenizer, discord_loop, channel_queue):
        self.history.add_message(Message(self.message.id, self.prompt, "user"))
        history = self.view.history.to_tokenizer(limit=self.message.id)
        history = tokenizer.history_to_tokens(history, cutoff=self.cutoff)
        threads = []
        for i in self.idxs:
            threads.append(threading.Thread(target=run_handler, args=[i, engine, history, self.view]))
        for i in threads:
            i.start()
            time.sleep(0.02)
        for i in threads:
            i.join()
