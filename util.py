from dotenv import load_dotenv
import nextcord as discord
import os
import torch

def init():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    dev_mode = True if os.getenv('DEV_MODE') == "1" else False
    dante_id = int(os.getenv('DANTE_ID'))
    intents = discord.Intents.all()
    client = discord.AutoShardedClient(intents=intents)
    discord.opus._load_default()
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return client, token, dev_mode, dante_id

def get_path(path_type="maw", data_type = "history", char_id=None, server_id=None):
    if path_type == "maw":
        if char_id != None:
            if data_type == "history":
                # Here, char_id is a channel
                if server_id == None:
                    return "data/dms/" + str(char_id) + "/history.txt"
                else:
                    return "data/servers/" + str(server_id) + "/" + str(char_id) + "/history.txt"
            else: return None
        else: return None
    elif path_type == "char":
        if char_id != None and server_id != None:
            if data_type == "history":
                return "data/servers/" + str(server_id) + "/char/" + str(char_id) + "/history.txt"
            if data_type == "config":
                return "data/server/" + str(server_id) + "/char/" + str(char_id) + "config.txt"
    else: return None

def get_all_chars(server_id):
    chars = []
    try:
        for i in os.listdir("data/server/" + str(server_id) + "/char/"):
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
        if owner.id == user.id:
            return True
        else:
            return False
    else:
        return True

def perm_check(channel, user, perm):
    if perm == "send":
        return channel.permissions_for(user).send_messages

def get_hook(channel, hooks, loop, client_id):
    try:
        hook = hooks[channel.id]
    except:
        all_hooks = asyncio.run_coroutine_threadsafe(coro=channel.webhooks(), loop=client.loop).result()
        for each_hook in all_hooks:
            if each_hook.user.id == client_id:
                asyncio.run_coroutine_threadsafe(coro=each_hook.delete(), loop=client.loop)
        hook = asyncio.run_coroutine_threadsafe(coro=channel.create_webhook(name="Character hook"), loop=client.loop).result()
        hooks[channel.id] = hook
    return hook
