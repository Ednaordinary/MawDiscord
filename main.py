from exllamav2.generator import ExLlamaV2Sampler
from exllamav2 import ExLlamaV2Cache_Q4
from queue import Queue
import nextcord as discord
import threading
import time
import os

from modeling.model_handler import Exl2ModelHandlerLazy
from modeling.exl2model import Exl2Loop
from modeling.tokenize import Tokenizer

from character.request import CharacterRequest
from character.history import History, Message
from character.defaults import MawPrompts

from util import init, get_path, get_all_chars, is_referring

client, token = init()

cutoff = 1024 * 15
model_loop = Exl2Loop()
model_handler = Exl2ModelHandlerLazy("dr1-32b-abliterated-exl2-4.0bpw-hb8", 1024 * 40, ExLlamaV2Cache_Q4, model_loop)
tokenizer = Tokenizer("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

character_queue = Queue()
handled_channels = {}

def channel_handler(channel_id):
    while len(handled_channels[channel_id]) > 0:
        model = None
        to_handle = handled_channels[channel_id]
        limiter = time.perf_counter()
        for i in model_handler.allocate(progress = True):
            if isinstance(i, tuple):
                for x in to_handle:
                    if i[0] == False:
                        x.update_progress("Waiting on " + i[1].strip(), client.loop)
                    else:
                        if limiter + 0.9 < time.perf_counter():
                            x.update_progress(str(int(100*(i[1][0] / i[1][1]))) + "%", client.loop)
                            limiter = time.perf_counter()
            else:
                model = i
        if model == None:
            print("Model was not properly allocated")
            return
        to_handle[0].handle(model, tokenizer, client.loop, character_queue)
        handled_channels[channel_id].pop(0)
    del handled_channels[channel_id]

def character_watcher():
    global character_queue
    while True:
        current = character_queue.get()
        if isinstance(current, CharacterRequest):
            if current.channel.id in handled_channels.keys():
                handled_channels[current.channel.id].append(current)
            else:
                handled_channels[current.channel.id] = [current]
                threading.Thread(target=channel_handler, args=[current.channel.id]).start()

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
    if maw_message:
        bot_message = await message.channel.send("...")
        history = History(get_path("maw", "history", char_id=message.channel.id, server_id=message.guild.id if message.guild else None), MawPrompts.default)
        prompt = str(message.author.nick or message.author.global_name or message.author.name or "User").strip() + " said: " + message.clean_content
        global character_queue
        character_queue.put(CharacterRequest(message, bot_message, history, prompt, cutoff, [], False))

@client.event
async def on_raw_message_edit(payload):
    if payload.guild_id:
        if os.path.exists(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)):
            history = History(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id), MawPrompts.default)
            try:
                channel = client.get_channel(payload.channel_id)
                message = await channel.fetch_message(payload.message_id)
            except:
                pass
            else:
                history.edit_message(Message(payload.message_id, message.content, None))

@client.event
async def on_raw_message_delete(payload):
    if payload.guild_id:
        if os.path.exists(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id)):
            history = History(get_path("maw", "history", char_id=payload.channel_id, server_id=payload.guild_id), MawPrompts.default)
            history.remove_message(payload.message_id)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(status=discord.Status.idle)

print("Starting up..")
threading.Thread(target=character_watcher).start()
client.run(token)
