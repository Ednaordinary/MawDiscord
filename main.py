from queue import Queue
import threading
import traceback
import asyncio
import random
import time
import os

import nextcord as discord

#from exllamav2 import ExLlamaV2Cache_Q4

#from modeling.model_handler.exl2 import Exl2ModelHandlerLazy
from modeling.model_handler.exl3 import Exl3ModelHandlerLazy, Exl3ModelHandler
from modeling.counter import RequestsCounter
from modeling.exl3model import Exl3Loop
from modeling.tokenize import Tokenizer

from character.views import ResetContextButton, CharacterModal, HookMessage
from character.request import CharacterRequest, RequestContext, ScrollRequest
from character.history import History, Message
from character.tools import Tool, DanteTool
from character.defaults import MawPrompts
from character.auto import SelfResponder
from character.config import Config

from util import init, get_path, get_all_chars, is_referring, dev_check, perm_check, get_hook, async_get_hook, get_history, relative_time, make_status, try_get_history, FakeHistObj

client, token, dev_mode, dante_id = init()
owner = None
tokens = 0
run_time = 0
histories = {}
hooks = {}
requests_counter = RequestsCounter("data/req.bin")

cutoff = 1024 * 32
model_loop = Exl3Loop()
quant = (4, 3)
#model_handler = Exl3ModelHandler("dr1-2-q3-8b-4bpw", 1024 * 256, quant, model_loop)
#tokenizer = Tokenizer("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
model_handler = Exl3ModelHandler("Qwen3-30B-A3B-exl3-3.5b", 1024 * 40, quant, model_loop)
tokenizer = Tokenizer("Qwen/Qwen3-30B-A3B")
#model_handler = Exl2ModelHandlerLazy("q3-30b-a3b-exl2", 1024 * 72, ExLlamaV2Cache_Q4, model_loop)
#tokenizer = Tokenizer("Qwen/Qwen3-30B-A3B")

character_queue = Queue()
handlers = 0

auto_responder = SelfResponder(character_queue, client, cutoff, tokenizer)

def handler(to_handle):
    try:
        global handlers, tokens, run_time
        handlers += 1
        activity = make_status(tokens, run_time, requests_counter.get())
        asyncio.run_coroutine_threadsafe(coro=client.change_presence(activity=activity, status=discord.Status.online), loop=client.loop)
        limiter = time.perf_counter()
        for i in model_handler.allocate(progress = True):
            if isinstance(i, tuple):
                if i[0] == False:
                    to_handle.update_progress("Waiting on " + i[1].strip(), client.loop)
                else:
                    if limiter + 1.5 < time.perf_counter():
                        to_handle.update_progress(str(int(100*(i[1][0] / i[1][1]))) + "%", client.loop)
                        limiter = time.perf_counter()
            else:
                model = i
        if model == None:
            print("Model was not properly allocated")
            return
        new_tokens = to_handle.handle(model, tokenizer, client.loop, character_queue)
        if new_tokens != None:
            tokens += new_tokens 
        current_run_time = model_handler.deallocate()
        if current_run_time is not None:
            run_time += current_run_time 
        print(tokens)
        print(run_time)
        handlers -= 1
        if handlers == 0:
            activity = make_status(tokens, run_time, requests_counter.get())
            asyncio.run_coroutine_threadsafe(coro=client.change_presence(status=discord.Status.idle, activity=activity), loop=client.loop)
    except:
        print(traceback.format_exc())

def character_watcher():
    global character_queue
    while True:
        current = character_queue.get()
        #if isinstance(current, CharacterRequest) or isinstance(current, ScrollRequest):
        threading.Thread(target=handler, args=[current]).start()

async def maw_send(message, auto=False):
    if perm_check(message.channel, message.guild, "send"):
        config_path = get_path("maw", "config", message)
        config = Config(config_path)
        if auto:
            bot_message = await message.reply(content="...")
        else:
            bot_message = await message.channel.send("...")
        history_path = get_path("maw", "history", message)
        history = get_history(history_path, histories, config)
        if isinstance(message.channel, discord.DMChannel):
            prompt = str(message.author.global_name).strip() + " said: " + message.clean_content
        else:
            prompt = str(message.author.nick or message.author.global_name or message.author.name or "User").strip() + " said: " + message.clean_content
        if auto:
            history = history.to_unwatched()
            msg_id = 9999999999999999999999999 # not preferable but does get a message to the bottom
            auto_prompt = MawPrompts.reply_focus + prompt
            history.add_message(Message(msg_id, auto_prompt, "user"))
        context = RequestContext(message, bot_message, history, prompt, False)
        tool = Tool()
        dante_tool = DanteTool(async_get_hook, dante_id, perm_check, message.channel, hooks, client.loop, client.user.id)
        tools = [tool, dante_tool] if not isinstance(message.channel, discord.DMChannel) else []
        global character_queue
        req_kwargs = {"context": context, "cutoff": cutoff, "tools": tools, "edit": False, "req_count": requests_counter, "queue": character_queue}
        character_queue.put(CharacterRequest(**req_kwargs))

def read():
    try:
        for i in auto_responder.should_respond():
            asyncio.run_coroutine_threadsafe(coro=maw_send(i, auto=True), loop=client.loop)
    except:
        print(traceback.format_exc())

@client.event
async def on_message(message):
    maw_message = False
    char_message = False
    if (message.type != discord.MessageType.default and message.type != discord.MessageType.reply) or message.author.id == client.user.id or message.webhook_id != None:
        return
    if isinstance(message.channel, discord.Thread) and message.channel.id in get_all_chars(message.guild.id or None):
        char_message = True
    elif "maw," in message.content.lower() or "<@" + str(client.user.id) + ">" in message.content or is_referring(message, client.user) or isinstance(message.channel, discord.DMChannel):
        maw_message = True
    if maw_message and dev_check(dev_mode, owner, message.author):
        await maw_send(message)
    elif maw_message:
        if perm_check(message.channel, message.guild, "send"):
            await message.channel.send("### >>> Maw is in dev mode. Please come back later.")
    if char_message and dev_check(dev_mode, owner, message.author):
        config_path = get_path("char", "config", message)
        config = Config(config_path)
        name = config.get()["name"]
        hook = await async_get_hook(message.channel.parent, hooks, client.user.id)
        bot_message = await hook.send("...", wait=True, username=name, thread=message.channel)
        bot_message = HookMessage(bot_message.id, hook, message.channel)
        history_path = get_path("char", "history", message)
        history = get_history(history_path, histories, None, char=True)
        prompt = message.clean_content
        context = RequestContext(message, bot_message, history, prompt, True)
        req_kwargs = {"context": context, "cutoff": cutoff, "tools": [], "edit": True, "req_count": requests_counter, "queue": character_queue}
        character_queue.put(CharacterRequest(**req_kwargs))
    elif char_message:
        if perm_check(message.channel, message.guild, "send"):
            await message.channel.send("### >>> Maw is in dev mode. Please come back later.")
    if not char_message:
        auto_responder.log_message(message)
        rand_value = random.randint(1, 100)
        print(rand_value)
        if rand_value == 1:
            print("Reading chat log")
            threading.Thread(target=read).start()

@client.event
async def on_raw_message_edit(payload):
    if payload.guild_id:
        history = try_get_history(payload.channel_id, payload.guild_id, histories)
        if history != None:
            try:
                if payload.cached_message and payload.cached_message.author.id == client.user.id:
                    return
                channel = client.get_channel(payload.channel_id)
                message = await channel.fetch_message(payload.message_id)
                if message.author.id == client.user.id or message.webhook_id != None:
                    return
            except:
                pass
            else:
                if isinstance(message.channel, discord.DMChannel):
                    prompt = str(message.author.global_name).strip() + " said: " + message.clean_content
                else:
                    if "char" in history.path:
                        prompt = message.clean_content
                    else:
                        prompt = str(message.author.nick or message.author.global_name or message.author.name or "User").strip() + " said: " + message.clean_content
                history.edit_message(Message(payload.message_id, prompt, "user"))

@client.event
async def on_raw_message_delete(payload):
    if payload.guild_id:
        history = try_get_history(payload.channel_id, payload.guild_id, histories)
        if history != None:
            history.remove_message(payload.message_id)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    activity = make_status(tokens, run_time, requests_counter.get())
    await client.change_presence(activity = activity, status=discord.Status.idle)
    global owner
    owner = await client.application_info()
    owner = owner.owner
    # hydrate data paths and message cache
    for guild in client.guilds:
        for channel in [x for x in guild.channels if isinstance(x, discord.TextChannel)]:
            obj = FakeHistObj(channel.id, guild.id)
            config_path = get_path("maw", "config", obj)
            config_file = Config(config_path)
            history_path = get_path("maw", "history", obj)
            history = get_history(history_path, histories, config_file)
            history.read_history()

@client.slash_command(description="Resets the context of Maw for the whole server (not including characters)")
async def reset(
        interaction: discord.Interaction,
):
    if os.path.exists(get_path("maw", "history", interaction)):
        config_path = get_path("maw", "config", interaction)
        config_file = Config(config_path)
        history_path = get_path("maw", "history", interaction)
        history = get_history(history_path, histories, config_file)
        history.read_history()
        ignored_ids = [0]
        if len([x for x in history.history if x.message_id not in ignored_ids]) > 0:
            view = ResetContextButton(history=history)
            await interaction.response.send_message(content="Are you sure? This will delete Maws memory in this server, not including characters or config.", view=view)
        else:
            await interaction.response.send_message(content="No context found to delete.")
    else:
        await interaction.response.send_message("No context found to clear.")

@client.slash_command(name="read", description="Forces an auto read (this is a debug action)")
async def read_cmd(
        interaction: discord.Interaction,
):
    await interaction.response.send_message("Reading...")
    print("Reading chat log")
    threading.Thread(target=read).start()

@client.slash_command(name="token")
async def token_root(
        interaction: discord.Interaction,
):
    pass


@token_root.subcommand(description="Get the current token count of any conversation.", name="count")
async def token_count(
        interaction: discord.Interaction,
):
    if os.path.exists(get_path("maw", "history", interaction)):
        config_path = get_path("maw", "config", interaction)
        config_file = Config(config_path)
        history_path = get_path("maw", "history", interaction)
        history = get_history(history_path, histories, config_file)
        history = history.to_tokenizer()
        history = tokenizer.history_to_tokens(history)
        length = history.shape[1]
        await interaction.response.send_message(str(length) + " tokens.")
    elif os.path.exists(get_path("char", "history", interaction)):
        history_path = get_path("char", "history", interaction)
        history = get_history(history_path, histories, None)
        history = history.to_tokenizer()
        history = tokenizer.history_to_tokens(history)
        length = history.shape[1]
        await interaction.response.send_message(str(length) + " tokens.", ephemeral=True)
    else:
        await interaction.response.send_message("0 tokens.")

@client.slash_command(description="Sends a form to make a character")
async def character(
        interaction: discord.Interaction,
):
    modal = CharacterModal(histories)
    await interaction.response.send_modal(modal)

@client.slash_command(name="config")
async def config_root(
        interaction: discord.Interaction,
):
    pass

@config_root.subcommand(name="personality")
async def personality_root(
        interaction: discord.Interaction,
):
    pass

@personality_root.subcommand(name="set", description="Sets channel wide personality")
async def personality_set(
        interaction: discord.Interaction,
        personality: str = discord.SlashOption(
            name="personality",
            required=True,
            description="Maws new personality",
            max_length=1500,
        ),
):
    config_path = get_path("maw", "config", interaction)
    config_file = Config(config_path)
    config = config_file.get()
    config["personality"] = personality.replace("\n", "\\n")
    config_file.write(config)
    history_path = get_path("maw", "history", interaction)
    get_history(history_path, histories, config_file)
    await interaction.response.send_message(content="Personality set.")

@personality_root.subcommand(name="reset")
async def personality_reset(
        interaction: discord.Interaction,
):
    config_path = get_path("maw", "config", interaction)
    config_file = Config(config_path)
    config = config_file.get()
    config["personality"] = MawPrompts.default_personality
    config_file.write(config)
    history_path = get_path("maw", "history", interaction)
    get_history(history_path, histories, config_file)
    await interaction.response.send_message(content="Personality reset.")

@config_root.subcommand(name="style", description="Sets channel wide typing style")
async def style_set(
        interaction: discord.Interaction,
        choice = discord.SlashOption(
            choices={"Short messages": "0", "Medium messages": "1", "Long messages": "2"}
        ),
):
    translation = [MawPrompts.style_short, MawPrompts.style_medium, MawPrompts.style_long]
    config_path = get_path("maw", "config", interaction)
    config_file = Config(config_path)
    config = config_file.get()
    config["style"] = translation[int(choice)]
    config_file.write(config)
    history_path = get_path("maw", "history", interaction)
    get_history(history_path, histories, config_file)
    await interaction.response.send_message(content="Style set.")

print("Starting up..")
threading.Thread(target=character_watcher).start()
client.run(token)
