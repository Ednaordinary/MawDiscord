import io
import os
import gc
import sys
import nextcord as discord
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import time
import torch
import threading
import asyncio
from typing import Optional

model_args = dict(max_new_tokens=256, use_cache=True, do_sample=True, max_matching_ngram_size=2, prompt_lookup_num_tokens=15) # PR isnt merged but including in readme

model_queue = []
hook_list = {} # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.
last_message = {}
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
all_tokens = 0
all_time = 0

class CharacterModal(discord.ui.Modal):
    def __init__(self, avatar):
        super().__init__(
            title="Make Character",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.avatar = avatar
        self.name = discord.ui.TextInput(
            label="Name",
            style=discord.TextInputStyle.short,
            placeholder="Name of the character. Ex: Maw",
            required=True,
            min_length=0,
            max_length=50,
        )
        self.add_item(self.name)

        self.description = discord.ui.TextInput(
            label="Description",
            style=discord.TextInputStyle.paragraph,
            placeholder="Describe the character from a third person view. Ex: You are graceful.",
            required=True,
            min_length=30,
            max_length=2000,
        )
        self.add_item(self.description)

        self.environment = discord.ui.TextInput(
            label="Starting environment",
            placeholder="The environment the character starts in. Ex: You in a large mansion.",
            style=discord.TextInputStyle.paragraph,
            required=True,
            min_length=0,
            max_length=1000,
        )
        
        self.add_item(self.environment)

        self.first_message = discord.ui.TextInput(
            label="First message",
            placeholder="The first message for the character to send. Can be blank.",
            style=discord.TextInputStyle.paragraph,
            required=False,
            min_length=0,
            max_length=1000,
        )
        self.add_item(self.first_message)
    async def callback(self, interaction: discord.Interaction) -> None:
        if self.description.value[-1] != ".": description = self.description.value + "."
        else: description = self.description.value
        prompt = "Your name is " + self.name.value + ". " + description + " To do an action, you surround actions in stars *like this*. You surround your dialogue in quotes " + '"like this"' + ". The person you are talking to may do the same with stars and quotes."
        response = prompt + " " + self.environment.value
        cut_value = 1900
        if len(response) > cut_value: response = response[:cut_value] + "(cont.)"
        else: response = response
        if self.avatar:
            avatar_image = await self.avatar.read()
            try:
                root = await interaction.send(response + "\n**Do not delete this message or the character will stop working**", file=discord.File(fp=io.BytesIO(avatar_image), filename="avatar.png"))
            except Exception as e:
                await interaction.send("Failed: " + str(repr(e)))
                return
        else:
            root = await interaction.send(response + "\n**Do not delete this message or the character will stop working**")
        try:
            try:
                root = await root.fetch()
                thread = await root.create_thread(name=self.name.value)
            except:
                thread = await root.create_thread(name=self.name.value)
        except:
            await root.edit("Thread could not be created (are you already in one?)")
        else:
            await thread.join()
            if self.avatar:
                config = MawCharacterConfig(prompt, self.environment.value, thread.id, "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/ids.txt", "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/history.txt", self.name.value, root.attachments[0].url)
            else:
                config = MawCharacterConfig(prompt, self.environment.value, thread.id, "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/ids.txt", "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/history.txt", self.name.value, None)
            make_maw_character("./characters/" + str(root.guild.id) + "/" + str(thread.id), config)
            character = MawCharacter(self.name.value, config, False)
            history = None
            if self.first_message.value != "":
                webhook = await get_webhook(root.channel)
                if self.avatar:
                    hook_message = await webhook.send(content=self.first_message.value, username=self.name.value, avatar_url=root.attachments[0].url, wait=True, thread=thread, view=EditMessageButton(character=character, user_message=None))
                else:
                    hook_message = await webhook.send(content=self.first_message.value, username=self.name.value, wait=True, thread=thread, view=EditMessageButton(character=character, user_message=None))
                history = [MawCharacterMessage(self.first_message.value, hook_message.id, "character")]
            if history:
                character.write_history(history)

class EditMessageModal(discord.ui.Modal):
    def __init__(self, original_content, character, user_message):
        super().__init__(
            title="Edit Message",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.original_content = original_content
        self.character = character
        self.user_message = user_message
        self.content = discord.ui.TextInput(
            label="Message",
            style=discord.TextInputStyle.paragraph,
            placeholder="New content of the message",
            default_value=self.original_content,
            required=True,
            min_length=1,
            max_length=2000,
        )
        self.add_item(self.content)
    async def callback(self, interaction: discord.Interaction) -> None:
        # This should not be called with maw, we don't allow editing maw message
        history = self.character.read_history()
        this_message = None
        for idx, message in enumerate(history):
            if int(message.message_id) == interaction.message.id:
                this_message = (idx, message)
        if this_message: #do not be destructive if stuff is weird
            history[this_message[0]] = MawCharacterMessage(self.content.value, interaction.message.id, "character")
            self.character.write_history(history)
            if this_message[0] == len(history) - 1 and not idx == 0: # if this is the latest message but not the first message, add a redo button
                view = EditAndRedoMessageButton(character=self.character, user_message=self.user_message)
            else:
                view = EditMessageButton(character=self.character, user_message=self.user_message)
            await interaction.response.edit_message(content=self.content.value, view=view)

class RedoMessageButton(discord.ui.View):
    def __init__(self, *,timeout=None, character, user_message):
        super().__init__(timeout=timeout)
        self.character = character
        self.user_message = user_message
    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary)
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.edit_message(content="...")
        history = self.character.read_history()
        try:
            self.character.write_history(history[:-2])
        except: pass
        model_queue.append(CharacterGen(self.user_message, interaction.message, self.character))

class EditAndRedoMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None, character, user_message):
        super().__init__(timeout=timeout)
        self.character = character
        self.user_message = user_message
    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary)
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_modal(EditMessageModal(interaction.message.content, self.character, self.user_message))
    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary)
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.edit_message(content="...")
        history = self.character.read_history()
        try:
            self.character.write_history(history[:-2])
        except: pass
        model_queue.append(CharacterGen(self.user_message, interaction.message, self.character))
    @discord.ui.button(label="Delete", style=discord.ButtonStyle.primary)
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        #Only characters use this class so delete is here
        history = self.character.read_history()
        message_idx = None
        for idx, message in enumerate(history):
            if int(message.message_id) == interaction.message.id:
                message_idx = idx
        if message_idx != None:
            history.pop(idx)
        self.character.write_history(history)
        await interaction.response.pong()
        await interaction.message.delete()

class EditMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None, character, user_message):
        super().__init__(timeout=timeout)
        self.character = character
        self.user_message = user_message
    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary)
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_modal(EditMessageModal(interaction.message.content, self.character, self.user_message))
    @discord.ui.button(label="Delete", style=discord.ButtonStyle.primary)
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        #Only characters use this class so delete is here
        history = self.character.read_history()
        message_idx = None
        for idx, message in enumerate(history):
            if int(message.message_id) == interaction.message.id:
                message_idx = idx 
        if message_idx != None:
            history.pop(idx)
        self.character.write_history(history)
        await interaction.response.pong()
        await interaction.message.delete()

class MawCharacterMessage:
    def __init__(self, content, message_id, role):
        self.content = content
        self.message_id = message_id
        self.role = role

class MawCharacterConfig:
    def __init__(self, system_prompt, environment_prompt, thread_id, ids_path, history_path, name, avatar):
        self.system_prompt = system_prompt
        self.environment_prompt = environment_prompt
        self.thread_id = thread_id
        self.ids_path = ids_path
        self.history_path = history_path
        self.name = name
        self.avatar = avatar

class MawCharacter:
    def __init__(self, name, config, maw):
        self.ids_path = config.ids_path
        self.history_path = config.history_path
        self.name = name
        self.config = config
        self.maw = maw # Is this maw or a character
    def write_history(self, history):
        with open(self.history_path, "w") as history_file:
            with open(self.ids_path, "w") as ids_file:
                for message in history:
                    history_file.write(message.content.replace("\n", r"\\n") + "\n")
                    role_prefix = "u" if message.role == "user" else "c"
                    ids_file.write(role_prefix + str(message.message_id)+"\n")
    def read_history(self):
        history = []
        if os.path.isfile(self.history_path):
            with open(self.history_path, "r") as history_file:
                with open(self.ids_path, "r") as ids_file:
                    history_lines, ids = history_file.readlines(), ids_file.readlines()
                    for idx, message in enumerate(history_lines):
                        role = "user" if ids[idx][:1] == "u" else "character"
                        message_id = ids[idx][1:-1] # For simplicity with maw vs character redo, this is a string
                        history.append(MawCharacterMessage(message[:-1].replace(r"\\n", "\n"), message_id, role))
        return history

class CharacterGen:
    def __init__(self, user_message, character_message, character):
        self.user_message = user_message
        self.character_message = character_message
        self.character = character

def make_maw_character(path, config):
    os.makedirs(path, exist_ok=True)
    with open(path + "/config.txt", "w") as config_file:
        if config.thread_id: config_file.write(str(config.thread_id).replace("\n", r"\\n") + "\n")
        else: config_file.write("0\n")
        config_file.write(str(config.system_prompt.replace("\n", r"\\n")) + "\n")
        config_file.write(str(config.environment_prompt.replace("\n", r"\\n")) + "\n")
        config_file.write(str(config.name.replace("\n", r"\\n")) + "\n")
        if config.avatar: config_file.write(str(config.avatar) + "\n")

def read_config(path):
    with open(path + "/config.txt", "r") as config_file:
        lines = config_file.readlines()
    if len(lines) > 4:
        return MawCharacterConfig(lines[1].replace(r"\\n", "\n"), lines[2].replace(r"\\n", "\n"), int(lines[0]), path + "/ids.txt", path + "/history.txt", lines[3], lines[4])
    else:
        return MawCharacterConfig(lines[1].replace(r"\\n", "\n"), lines[2].replace(r"\\n", "\n"), int(lines[0]), path + "/ids.txt", path + "/history.txt", lines[3], None)

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
            token_length += llama_message.shape[1]
        else:
            break
    if config.environment_prompt != "":
        environment_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": config.environment_prompt}], tokenize=True, return_tensors='pt', add_generation_prompt=False)
        if token_length + llama_message.shape[1] < (7200 - system_prompt.shape[1]):
            llama.append(environment_prompt)
            token_length += llama_message.shape[1]
    llama.append(system_prompt)
    llama.reverse()
    history.reverse() # this was inplace so it needs to be flipped back
    llama = torch.cat(llama, 1)
    print(token_length, llama.shape)
    return llama

async def edit_add_redobutton(message, content, character, user_message):
    #views cannot be crafted outside of an event loop
    await message.edit(content, view=RedoMessageButton(character=character, user_message=user_message))

async def edit_add_editredobutton(hook, message, content, character, user_message, thread_id):
    #views cannot be crafted outside of an event loop
    await hook.edit_message(content=content, message_id=message.id, thread=thread_id, view=EditAndRedoMessageButton(character=character, user_message=user_message))

async def get_webhook(channel):
    # unfortunately, we have to redo hooks every bot start to use views. This is because of how ownership works
    try:
        hook = hook_list[channel.id]
    except:
        all_hooks = await channel.webhooks()
        for each_hook in all_hooks:
            if each_hook.user == client.user:
                await each_hook.delete()
        hook = await channel.create_webhook(name="Character hook")
        hook_list[channel.id] = hook
    return hook

async def temp_edit(message_id, thread_id, content, channel_id):
    await hook_list[channel_id].edit_message(message_id=message_id, content=content, thread=thread_id)

def message_updater(message, streamer, character, user_message, thread_id, channel):
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
                asyncio.run_coroutine_threadsafe(coro=temp_edit(message.id, thread_id, full_text, channel.id), loop=client.loop)
    if character.maw:
        asyncio.run_coroutine_threadsafe(coro=edit_add_redobutton(message, full_text, character, user_message), loop=client.loop)
    else:
        asyncio.run_coroutine_threadsafe(coro=edit_add_editredobutton(hook_list[channel.id], message, full_text, character, user_message, thread_id), loop=client.loop)

def watcher():
    global all_tokens
    global all_time
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
            if all_tokens != 0:
                asyncio.run_coroutine_threadsafe(coro=client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.online), loop=client.loop)
            current_gen = model_queue[0]
            history = current_gen.character.read_history()
            history.append(MawCharacterMessage(current_gen.user_message.content, str(current_gen.user_message.id), "user"))
            current_gen.character.write_history(history) # if message is edited or deleted during generation, it needs to be reflected
            if model == None:
                model = AutoModelForCausalLM.from_pretrained(
                    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                    load_in_8bit=False,
                )
            gc.collect()
            torch.cuda.empty_cache()
            model_input = history_to_llama(history, tokenizer, current_gen.character.config)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            if current_gen.character.maw: thread_id, channel = None, current_gen.character_message.channel
            else: thread_id, channel = current_gen.user_message.channel, current_gen.user_message.channel.parent
            streamer_thread = threading.Thread(target=message_updater, args=[current_gen.character_message, streamer, current_gen.character, current_gen.user_message, thread_id, channel])
            streamer_thread.start()
            start_time = time.time()
            response = model.generate(input_ids=model_input.to('cuda'), **model_args, streamer=streamer, eos_token_id=stop_token)
            all_tokens += len(response[0][model_input.shape[1]:])
            all_time += time.time() - start_time
            asyncio.run_coroutine_threadsafe(coro=client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(round(len(response[0][model_input.shape[1]:]) / (time.time() - start_time), 2)) + " tps | " + str(round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.idle), loop=client.loop)
            decoded_response = tokenizer.decode(response[0][model_input.shape[1]:], skip_special_tokens=True)
            if current_gen.character.maw: history.append(MawCharacterMessage(decoded_response, (str(current_gen.character_message.id) + "-" + str(current_gen.character_message.channel.id)), "character"))
            else: history.append(MawCharacterMessage(decoded_response, current_gen.character_message.id, "character"))
            current_gen.character.write_history(history)
            del response, decoded_response, model_input
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
    character_response = False
    if "maw," in message.content.lower() and not r"\end" in message.content.lower() and not "/end" in message.content.lower(): maw_response = True
    try:
        last_message[message.guild.id]
    except:
        print("saving last message")
        last_message[message.guild.id] = message
    else:
        if last_message[message.guild.id].author.id == client.user.id and message.author.id != client.user.id and last_message[message.guild.id].channel == message.channel and not r"\end" in message.content and not "/end" in message.content:
            maw_response = True
    if os.path.isdir("./characters/" + str(message.guild.id) + "/" + str(message.channel.id)):
        character_response = True
        maw_response = False
    if message.author.bot:
        character_response = False
        maw_response = False
    if type(message.channel) == discord.Thread:
        maw_response = False # too much weird stuff with how threads are handled right now
    if maw_response:
        last_message[message.guild.id] = message
        maw_message = await message.channel.send("...")
        old_message_id = None
        if os.path.isdir("./servers/" + str(message.guild.id)):
            config = read_config("./servers/" + str(message.guild.id))
            character = MawCharacter("Maw", config, True)
            if os.path.isfile("./servers/" + str(message.guild.id) + "/history.txt"):
                history = character.read_history()
                try:
                    old_message_id = (int(history[-1].message_id.split("-")[-2]), int(history[-1].message_id.split("-")[-1]))
                except:
                    pass
        else:
            system_prompt = "You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna."
            config = MawCharacterConfig(system_prompt, "", None, "./servers/" + str(message.guild.id) + "/ids.txt", "./servers/" + str(message.guild.id) + "/history.txt", "Maw", None)
            make_maw_character("./servers/" + str(message.guild.id), config)
            character = MawCharacter("Maw", config, True)
        model_queue.append(CharacterGen(message, maw_message, character))
        if old_message_id:
            channel = client.get_channel(old_message_id[1])
            old_message = await channel.fetch_message(old_message_id[0])
            await old_message.edit(old_message.content, view=None)
    if character_response:
        hook = await get_webhook(message.channel.parent)
        config = read_config("./characters/" + str(message.guild.id) + "/" + str(message.channel.id))
        if config.avatar:
            character_message = await hook.send(content="...", username=config.name, wait=True, thread=message.channel, avatar_url=config.avatar)
        else:
            character_message = await hook.send(content="...", username=config.name, wait=True, thread=message.channel)
        character = MawCharacter(config.name, config, False)
        old_message_id = None
        if os.path.isfile("./characters/" + str(message.guild.id) + "/" + str(message.channel.id) + "/history.txt"):
            history = character.read_history()
            try:
                old_message_id = int(history[-1].message_id)
            except: pass
            try:
                user_message = history[-2]
            except:
                user_message = None
        model_queue.append(CharacterGen(message, character_message, character))
        if old_message_id:
            try:
                await hook.edit_message(message_id=old_message_id, view=EditMessageButton(character=character, user_message=user_message), thread=message.channel)
            except: pass # isn't really needed but I don't like random error messages in my console

@client.event
async def on_raw_message_edit(payload):
    payload.message_id
    channel = client.get_channel(payload.channel_id)
    try: payload.data["content"]
    except: pass
    else:
        if isinstance(channel, discord.Thread) and os.path.exists("./characters/" + str(channel.guild.id) + "/" + str(channel.id) + "/"):
            config = read_config("./characters/" + str(channel.guild.id) + "/" + str(channel.id))
            character = MawCharacter(config.name, config, False)
            new_message = await channel.fetch_message(payload.message_id)
            if not new_message.author.bot:
                history = character.read_history()
                message_idx = None
                for idx, message in enumerate(history):
                    if int(message.message_id) == payload.message_id:
                        message_idx = idx
                        break
                if message_idx != None:
                    history[message_idx] = MawCharacterMessage(payload.data["content"], history[message_idx].message_id, history[message_idx].role)
                character.write_history(history)

@client.event
async def on_raw_message_delete(payload):
    payload.message_id
    channel = client.get_channel(payload.channel_id)
    if isinstance(channel, discord.Thread) and os.path.exists("./characters/" + str(channel.guild.id) + "/" + str(channel.id) + "/"):
        config = read_config("./characters/" + str(channel.guild.id) + "/" + str(channel.id))
        character = MawCharacter(config.name, config, False)
        message_idx = None
        history = character.read_history()
        for idx, message in enumerate(history):
            if int(message.message_id) == payload.message_id:
                message_idx = idx
                break
        if message_idx != None: history.pop(message_idx)
        character.write_history(history)
        try:
            edit_message = history[-1]
        except:
            pass
        else:
            user_message=None
            try:
                user_message = await channel.fetch_message(history[-2].message_id)
            except:
                pass
            if edit_message.role == "character":
                edit_message = await channel.fetch_message(edit_message.message_id)
                hook = await get_webhook(channel.parent)
                await hook.edit_message(message_id=edit_message.id, thread=channel, view=EditAndRedoMessageButton(character=character, user_message=user_message))

@client.slash_command(description="Sends a form to make a character")
async def character(
        interaction: discord.Interaction,
        avatar: Optional[discord.Attachment] = discord.SlashOption(
            name="avatar",
            required=False,
            description="An avatar for your character. Must be jpg or png",
        ),
):
    if avatar and not avatar.content_type == "image/jpeg" and not avatar.content_type == "image/png":
        await interaction.response.send_message("Avatar is not png or jpg. Please try again")
    else:
        modal = CharacterModal(avatar)
        await interaction.response.send_modal(modal)

threading.Thread(target=watcher).start()
client.run(TOKEN)
