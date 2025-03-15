from dotenv import load_dotenv
import nextcord as discord
import os
import torch

def init():
    load_dotenv()
    token = os.getenv('DISCORD_TOKEN')
    intents = discord.Intents.all()
    client = discord.AutoShardedClient(intents=intents)
    discord.opus._load_default()
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["TOKENIZERS_PARALLELISM"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return client, token

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
    for i in os.listdirs("data/server/" + str(server_id) + "/char/"):
        try:
            chars.append(int(i))
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
