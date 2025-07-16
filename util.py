from dotenv import load_dotenv
import nextcord as discord
import asyncio
import torch
import os

from character.history import JsonHistory
from character.defaults import MawPrompts

def init():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    dev_mode = True if os.getenv('DEV_MODE') == "1" else False
    dante_id = int(os.getenv('DANTE_ID'))
    intents = discord.Intents.all()
    client = discord.Client(intents=intents)
    discord.opus._load_default()
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return client, token, dev_mode, dante_id

def get_path(path_type="maw", data_type = "history", obj=None):
    char_id = obj.channel.id
    try:
        server_id = obj.guild.id
    except:
        server_id = None
    if path_type == "maw":
        if char_id != None:
            if data_type == "history":
                # Here, char_id is a channel
                if server_id == None:
                    return "data/dms/" + str(char_id) + "/history.json"
                else:
                    return "data/servers/" + str(server_id) + "/" + str(char_id) + "/history.json"
            if data_type == "config":
                if server_id == None:
                    return "data/dms/" + str(char_id) + "/config.json"
                else:return "data/servers/" + str(server_id) + "/" + str(char_id) + "/config.json"
                    
            else: return None
        else: return None
    elif path_type == "char":
        if char_id != None and server_id != None:
            if data_type == "history":
                return "data/servers/" + str(server_id) + "/char/" + str(char_id) + "/history.json"
            if data_type == "config":
                return "data/servers/" + str(server_id) + "/char/" + str(char_id) + "/config.json"
    else: return None

def get_all_chars(server_id):
    chars = []
    try:
        for i in os.listdir("data/servers/" + str(server_id) + "/char/"):
            try:
                chars.append(int(i))
            except:
                pass
    except:
        pass
    return chars

def is_referring(message, user):
    try:
        if message.reference.cached_message.author.id == user.id:
            return True
        else:
            return False
    except:
        return False

def dev_check(dev_mode, owner, user):
    if dev_mode:
        if owner != None and owner.id == user.id:
            return True
        else:
            return False
    else:
        return True

def perm_check(channel, guild, perm):
    if guild == None:
        return True
    user = guild.me
    if perm == "send":
        return channel.permissions_for(user).send_messages

def get_hook(channel, hooks, loop, client_id):
    try:
        hook = hooks[channel.id]
    except:
        all_hooks = asyncio.run_coroutine_threadsafe(coro=channel.webhooks(), loop=loop).result(timeout=2)
        for each_hook in all_hooks:
            if each_hook.user.id == client_id:
                asyncio.run_coroutine_threadsafe(coro=each_hook.delete(), loop=loop)
        hook = asyncio.run_coroutine_threadsafe(coro=channel.create_webhook(name="Character hook"), loop=loop).result(timeout=3)
        hooks[channel.id] = hook
    return hook

async def async_get_hook(channel, hooks, client_id):
    try:
        hook = hooks[channel.id]
    except:
        print("getting current hooks")
        all_hooks = await channel.webhooks()
        for each_hook in all_hooks:
            if each_hook.user.id == client_id:
                print("deleting an old hook")
                await each_hook.delete()
        print("making a new hook")
        hook = await channel.create_webhook(name="Character hook")
        hooks[channel.id] = hook
    return hook

def get_history(path, histories, config, char=False):
    try:
        history = histories[path]
        if config != None:
            history.sys = config if (char or config == None) else update_sys(config, char)
            history.renew_sys()
    except:
        history = JsonHistory(path, config if (char or config == None) else update_sys(config, char))
        histories[path] = history
    return history

def update_sys(config_file, char):
    sys = MawPrompts.default
    config = config_file.get()
    if not "personality" in config.keys():
        config["personality"] = MawPrompts.default_personality
    if not "style" in config.keys():
        config["style"] = MawPrompts.style_short
    config_file.write(config)
    sys += "\n" + str(config["personality"])
    sys += "\n" + str(config["style"])
    return sys

def relative_time(time):
    discord_epoch = 1420070400000
    epoch_offset = time >> 22
    epoch_time = (discord_epoch + epoch_offset) / 1000
    print(epoch_time)
    return "<t:" + str(int(epoch_time)) + ":R>"

def make_status(tokens, run_time, requests):
    if run_time > 0:
        activity = discord.Activity(type=discord.ActivityType.watching, name="at " + str(int(tokens / run_time)) + " avg tps | " + str(requests) + " requests handled")
    else:
        activity = discord.Activity(type=discord.ActivityType.watching, name="with " + str(requests) + " requests handled")
    return activity

class NestedObj:
    def __init__(self):
        pass

class FakeHistObj:
    def __init__(self, char_id, server_id):
        self.guild = NestedObj()
        self.guild.id = server_id
        self.channel = NestedObj()
        self.channel.id = char_id

def try_get_history(char_id, server_id, histories):
    history_path = None
    if os.path.exists(get_path("maw", "history", FakeHistObj(char_id, server_id))):
        history_path = get_path("maw", "history", FakeHistObj(char_id, server_id))
    elif os.path.exists(get_path("char", "history", FakeHistObj(char_id, server_id))):
        history_path = get_path("char", "history", FakeHistObj(char_id, server_id))
    if history_path != None:
        return get_history(history_path, histories, None)
    else: return None
