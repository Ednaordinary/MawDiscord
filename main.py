import os
import gc
import nextcord as discord
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import torch
import threading

model_args = dict(max_new_tokens=768, use_cache=True, do_sample=True) #, max_matching_ngram_size=2, prompt_lookup_num_tokens=15) # waiting for PR in transformers to be merged

model_queue = []
hooks = {} # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.

class MawCharacterMessage:
    def __init__(self, content, message_id, role):
        self.content = content
        self.message_id = message_id
        self.role = role

class MawCharacterConfig:
    def __init__(self, system_prompt, environment_prompt, thread_id):
        self.system_prompt = system_prompt
        self.environment_prompt = environment_prompt
        self.thread_id = thread_id

class MawCharacter:
    def __init__(self, ids_path, history_path, name, config, maw):
        self.ids_path = ids_path
        self.history_path = history_path
        self.name = name
        self.config = config
        self.maw = maw # Is this maw or a character
    def write_history(history):
        with open(self.history_path, "w"), open(self.ids_path, "w") as history_file, ids_file:
            for message in history:
                history_file.write(message.content.replace("\n", "\\n") + "\n")
                role_prefix = "u" if message.role == "user" else ("c" if message.role == "character" else "s" )
                ids_file.write(role_prefix + str(message.message_id))
    def read_history():
        if os.path.isfile(self.history_path, "r"):
            with open(self.history_path, "r"), open(self.ids_path, "r") as history_file, ids_file:
                history_lines, ids = history_file.readlines(), ids.readlines()
                history = []
                for idx, message in enumerate(history):
                    try: # if they don't line up, stuff will break
                        role = "user" if ids[idx][:1] == "u" else ("character" if ids[idx][:1] == "c" else "system")
                        message_id = ids[idx][1:]
                        history.append(MawCharacterMessage(message.replace("\\n", "\n"), message_id, role))
                return history

def make_maw_character(path, config):
    os.makedirs(path, exist_ok=True)
    with open(path + "/config.txt", "w") as config_file:
        config_file.write(str(config.thread_id.replace("\n", "\\n")) + "\n")
        config_file.write(str(config.system_prompt.replace("\n", "\\n")) + "\n")
        config_file.write(str(config.environment_prompt.replace("\n", "\\n")) + "\n")

def read_config(path):
    with open(path + "/config.txt", "r") as config_file:
        lines = config_file.readlines()
    return MawCharacterConfig(lines[1].replace("\\n", "\n"), lines[2].replace("\\n", "\n"), int(lines[0]))

class CharacterGen:
    def __init__(self, user_message, character_message, character):
        self.user_message = user_message
        self.character_message = character_message
        self.character = character

def history_to_llama(history, tokenizer, config):
    llama = []
    token_length = 0
    system_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": config.system_prompt}], tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    for message in history:
        role = "assistant" if message.role == "character" else message.role
        llama_message = [{"role": role, "content": message.content}]
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        if token_length + llama_message.shape[1] < (7200 - system_prompt.shape[1]):
            llama.append(llama_message)
        else:
            break
    llama.append(system_prompt)
    llama.reverse()
    llama = torch.cat(llama, 1)
    return llama

def message_updater(message, streamer, character):
    full_text = ""
    limiter = time.time()
    for text in streamer:
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
        if model == None:
            model = AutoModelForCausalLM.from_pretrained(
                "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                device="cuda",
            )
        gc.collect()
        torch.cuda.empty_cache()
        current_gen = model_queue[0]
        history = current_gen.character.read_history()
        history.append(MawCharacterMessage(current_gen.user_message.content, current_gen.user_message.id, "user"))
        history = history_to_llama(history, tokenizer, current_gen.character.config)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer_thread = message_updater(current_gen.user_message, streamer, current_gen.character)
        streamer_thread.start()
        response = model.generate(**kwargs, streamer=streamer, eos_token_id=stop_token)
        gc.collect()
        torch.cuda.empty_cache()
        model_queue.pop(0)

@client.event
def on_message():
    

client.run(TOKEN)
