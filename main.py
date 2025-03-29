from exllamav2.generator import ExLlamaV2Sampler
from exllamav2 import ExLlamaV2Cache_Q4
from queue import Queue
import nextcord as discord
import threading
import asyncio
import time
import os

from modeling.model_handler import Exl2ModelHandlerLazy
from modeling.exl2model import Exl2Loop
from modeling.tokenize import Tokenizer

from character.request import CharacterRequest
from character.history import History, Message
from character.defaults import MawPrompts
from character.tools import Tool, DanteTool

from util import init, get_path, get_all_chars, is_referring, dev_check, perm_check, get_hook, get_history

client, token, dev_mode, dante_id = init()
owner = None
tokens = 0
run_time = 0
histories = {}
hooks = {}

cutoff = 1024 * 15
model_loop = Exl2Loop()
model_handler = Exl2ModelHandlerLazy("dr1-32b-abliterated-exl2-4.0bpw-hb8", 1024 * 35, ExLlamaV2Cache_Q4, model_loop)
tokenizer = Tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

character_queue = Queue()
handlers = 0

def handler(to_handle):
    global handlers, tokens, run_time
    handlers += 1
    if run_time != 0:
        activity = discord.Activity(type=discord.ActivityType.watching, name="at " + str(int(tokens / run_time)) + " avg tps")
        asyncio.run_coroutine_threadsafe(coro=client.change_presence(activity=activity, status=discord.Status.online), loop=client.loop)
    else:
        asyncio.run_coroutine_threadsafe(coro=client.change_presence(status=discord.Status.online), loop=client.loop)
    limiter = time.perf_counter()
    for i in model_handler.allocate(progress = True):
        if isinstance(i, tuple):
            if i[0] == False:
                to_handle.update_progress("Waiting on " + i[1].strip(), client.loop)
            else:
                if limiter + 0.9 < time.perf_counter():
                    to_handle.update_progress(str(int(100*(i[1][0] / i[1][1]))) + "%", client.loop)
                    limiter = time.perf_counter()
        else:
            model = i
    if model == None:
        print("Model was not properly allocated")
        return
    tokens += to_handle.handle(model, tokenizer, client.loop, character_queue)
    run_time += model_handler.deallocate()
    handlers -= 1
    if handlers == 0:
        if run_time != 0:
            activity = discord.Activity(type=discord.ActivityType.watching, name="at " + str(int(tokens / run_time)) + " avg tps")
        asyncio.run_coroutine_threadsafe(coro=client.change_presence(status=discord.Status.idle, activity=activity), loop=client.loop)

def character_watcher():
    global character_queue
    while True:
        current = character_queue.get()
        if isinstance(current, CharacterRequest):
            threading.Thread(target=handler, args=[current]).start()

@client.event
async def on_message(message):
    maw_message = False
    char_message = False
    if message.type != discord.MessageType.default and message.type != discord.MessageType.reply or message.author.id == client.user.id or message.webhook_id != None:
        return
    if isinstance(message.channel, discord.Thread) and message.channel.id in get_all_chars(message.guild.id):
        char_message = True
    elif "maw," in message.content.lower() or "<@" + str(client.user.id) + ">" in message.content or is_referring(message, client.user):
        maw_message = True
    if maw_message and dev_check(dev_mode, owner, message.author):
        if perm_check(message.channel, message.guild.me, "send"):
            bot_message = await message.channel.send("...")
            history_path = get_path("maw", "history", char_id=message.channel.id, server_id=message.guild.id)
            history = get_history(history_path, histories, MawPrompts.default)
            prompt = str(message.author.nick or message.author.global_name or message.author.name or "User").strip() + " said: " + message.clean_content
            global character_queue
            tool = Tool()
            dante_tool = DanteTool(get_hook, dante_id, perm_check, message.channel, hooks, client.loop, client.user.id)
            character_queue.put(CharacterRequest(message, bot_message, history, prompt, cutoff, [tool, dante_tool], False))
    elif maw_message:
        if perm_check(message.channel, message.guild.me, "send"):
            await message.channel.send("### >>> Maw is in dev mode. Please come back later.")

@client.event
async def on_raw_message_edit(payload):
    if payload.guild_id:
        if os.path.exists(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)):
            history_path = get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)
            history = get_history(history_path, histories, MawPrompts.default)
            try:
                if payload.cached_message and payload.cached_message.author.id == client.user.id:
                    return
                channel = client.get_channel(payload.channel_id)
                message = await channel.fetch_message(payload.message_id)
                if message.author.id == client.user.id:
                    pass
            except:
                pass
            else:
                history.edit_message(Message(payload.message_id, message.content, None))

@client.event
async def on_raw_message_delete(payload):
    if payload.guild_id:
        if os.path.exists(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)):
            history_path = get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)
            history = get_history(history_path, histories, MawPrompts.default)
            history.remove_message(payload.message_id)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(status=discord.Status.idle)
    global owner
    owner = await client.application_info()
    owner = owner.owner

print("Starting up..")
threading.Thread(target=character_watcher).start()
client.run(token)
