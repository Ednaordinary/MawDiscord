import gc
import nextcord as discord
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import torch
import threading

model_queue = []

class MawCharacterMessage:
    def __init__(self, content, message_id, role):
        self.content = content
        self.message_id = message_id
        self.role = role

class MawCharacter:
    def __init__(self, ids_path, history_path, name, system_prompt, environment_prompt):
        self.ids_path = ids_path
        self.history_path = history_path
        self.name = name
        self.system_prompt = system_prompt
        self.environment_prompt = environment_prompt
    def write_history(history):
        with open(self.history_path, "w"), open(self.ids_path, "w") as history_file, ids_file:
            for message in history:
                history_file.write(message.content.replace("\n", "\\n") + "\n")
                role_prefix = "u" if message.role == "user" else ("c" if message.role == "character" else "s" )
                ids_file.write(role_prefix + str(message.message_id))
    def read_history(history):
        with open(self.history_path, "r"), open(self.ids_path, "r") as history_file, ids_file:
            history_lines, ids = history_file.readlines(), ids.readlines()
            history = []
            for idx, message in enumerate(history):
                try: # if they don't line up, stuff will break
                    role = "user" if ids[idx][:1] == "u" else ("character" if ids[idx][:1] == "c" else "system")
                    message_id = ids[idx][1:]
                    history.append(MawCharacterMessage(message.replace("\\n", "\n"), message_id, role))
            return history

class CharacterGen(self, user_message, character_message)

def watcher():
    model = None
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
            tokenizer = AutoTokenizer.from_pretrained(
                "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                device="cuda",
                truncation_side='left'
            )
        gc.collect()
        torch.cuda.empty_cache()
        current_gen = model_queue[0]
        message = 
client.run(TOKEN)
