import os
import gc
import nextcord as discord
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import time
import torch
import threading
import asyncio

model_args = dict(max_new_tokens=768, use_cache=True, do_sample=True) #, max_matching_ngram_size=2, prompt_lookup_num_tokens=15) # waiting for PR in transformers to be merged

model_queue = []
hooks = {} # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.
last_message = {}
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)

class MawCharacterMessage:
    def __init__(self, content, message_id, role):
        self.content = content
        self.message_id = message_id
        self.role = role

class MawCharacterConfig:
    def __init__(self, system_prompt, environment_prompt, thread_id, ids_path, history_path):
        self.system_prompt = system_prompt
        self.environment_prompt = environment_prompt
        self.thread_id = thread_id
        self.ids_path = ids_path
        self.history_path = history_path

class MawCharacter:
    def __init__(self, name, config, maw):
        self.ids_path = config.ids_path
        self.history_path = config.history_path
        self.name = name
        self.config = config
        self.maw = maw # Is this maw or a character
    def write_history(self, history):
        with (open(self.history_path, "w"), open(self.ids_path, "w")) as (history_file, ids_file):
            for message in history:
                history_file.write(message.content.replace("\n", "\\n") + "\n")
                role_prefix = "u" if message.role == "user" else ("c" if message.role == "character" else "s" )
                ids_file.write(role_prefix + str(message.message_id))
    def read_history(self):
        history = []
        if os.path.isfile(self.history_path):
            with (open(self.history_path, "r"), open(self.ids_path, "r")) as (history_file, ids_file):
                history_lines, ids = history_file.readlines(), ids.readlines()
                for idx, message in enumerate(history):
                    try: # if they don't line up, stuff will break
                        role = "user" if ids[idx][:1] == "u" else ("character" if ids[idx][:1] == "c" else "system")
                        message_id = ids[idx][1:]
                        history.append(MawCharacterMessage(message.replace("\\n", "\n"), message_id, role))
                    except: pass
        return history

class CharacterGen:
    def __init__(self, user_message, character_message, character):
        self.user_message = user_message
        self.character_message = character_message
        self.character = character

def make_maw_character(path, config):
    os.makedirs(path, exist_ok=True)
    with open(path + "/config.txt", "w") as config_file:
        if config.thread_id: config_file.write(str(config.thread_id.replace("\n", "\\n")) + "\n")
        else: config_file.write("0\n")
        config_file.write(str(config.system_prompt.replace("\n", "\\n")) + "\n")
        config_file.write(str(config.environment_prompt.replace("\n", "\\n")) + "\n")

def read_config(path):
    with open(path + "/config.txt", "r") as config_file:
        lines = config_file.readlines()
    return MawCharacterConfig(lines[1].replace("\\n", "\n"), lines[2].replace("\\n", "\n"), int(lines[0]), path + "/ids.txt", path + "/history.txt")

def history_to_llama(history, tokenizer, config):
    llama = []
    token_length = 0
    system_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": config.system_prompt}], tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    for idx, message in enumerate(history):
        role = "assistant" if message.role == "character" else message.role
        llama_message = [{"role": role, "content": message.content}]
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt', add_generation_prompt=True if idx == 0 else False)
        if token_length + llama_message.shape[1] < (7200 - system_prompt.shape[1]):
            llama.append(llama_message)
        else:
            break
    if config.environment_prompt != "":
        environment_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": config.environment_prompt}], tokenize=True, return_tensors='pt', add_generation_prompt=False)
        if token_length + llama_message.shape[1] < (7200 - system_prompt.shape[1]):
            llama.append(environment_prompt)
    llama.append(system_prompt)
    llama.reverse()
    llama = torch.cat(llama, 1)
    return llama

def message_updater(message, streamer, character):
    full_text = ""
    limiter = time.time()
    for text in streamer:
        print(text, flush=True, end='')
        full_text = full_text + text
        if time.time() - limiter > 0.8:
            limiter = time.time()
            if character.maw:
                asyncio.run_coroutine_threadsafe(coro=message.edit(full_text), loop=client.loop)
            else:
                asyncio.run_coroutine_threadsafe(coro=hook[message.channel.parent.id].edit_message(message_id=message.id, content=full_text, thread=message.channel), loop=client.loop)
    if character.maw:
        asyncio.run_coroutine_threadsafe(coro=message.edit(full_text), loop=client.loop)
    else:
        asyncio.run_coroutine_threadsafe(coro=hook[message.channel.parent.id].edit_message(message_id=message.id, content=full_text, thread=message.channel), loop=client.loop)

def watcher():
    model = None
    tokenizer = AutoTokenizer.from_pretrained(
                "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
            ) # can just be kept loaded
    stop_token = tokenizer.encode("<|eot_id|>")
    while True:
        if model_queue == []:
            if model != None:
                model = None
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(0.01)
        else:
            if model == None:
                model = AutoModelForCausalLM.from_pretrained(
                    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )
            gc.collect()
            torch.cuda.empty_cache()
            current_gen = model_queue[0]
            history = current_gen.character.read_history()
            history.append(MawCharacterMessage(current_gen.user_message.content, current_gen.user_message.id, "user"))
            model_input = history_to_llama(history, tokenizer, current_gen.character.config)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            streamer_thread = threading.Thread(target=message_updater, args=[current_gen.character_message, streamer, current_gen.character])
            streamer_thread.start()
            response = model.generate(input_ids=model_input.to('cuda'), **model_args, streamer=streamer, eos_token_id=stop_token)
            history.append(MawCharacterMessage(response, current_gen.character_message.id, "character"))
            current_gen.character.write_history(history)
            gc.collect()
            torch.cuda.empty_cache()
            model_queue.pop(0)

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(status=discord.Status.idle)

@client.event
async def on_message(message):
    global model_queue
    global last_message
    maw_response = False
    if "Maw," in message.content and not r"\end" in message.content and not "/end" in message.content: maw_response = True
    if maw_response:
        maw_message = await message.channel.send("...")
        if os.path.isdir("./servers/" + str(message.guild.id)):
            config = read_config("./servers/" + str(message.guild.id))
            character = MawCharacter("Maw", config, True)
        else:
            system_prompt = "You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna."
            config = MawCharacterConfig(system_prompt, "", None, "./servers/" + str(message.guild.id) + "/ids.txt", "./servers/" + str(message.guild.id) + "/history.txt")
            make_maw_character("./servers/" + str(message.guild.id), config)
            character = MawCharacter("Maw", config, True)
        model_queue.append(CharacterGen(message, maw_message, character))

threading.Thread(target=watcher).start()
client.run(TOKEN)
