import io
import os
import gc
import re
import sys
import nextcord as discord
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, TextIteratorStreamer, AutoTokenizer
import time
import torch
import threading
import asyncio
from typing import Optional
import vram

model_args = dict(max_new_tokens=768, use_cache=True, do_sample=True, max_matching_ngram_size=2,
                  prompt_lookup_num_tokens=15, repetition_penalty=1.2)  # PR isnt merged but including in readme

model_queue = []
hook_list = {}  # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.
second_last_message = {}
last_message = {}
watched_avatars = []
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.Client(intents=intents)
all_tokens = 0
all_time = 0


# this class is only used by characters
class CharacterModal(discord.ui.Modal):
    def __init__(self, avatar, locked):
        super().__init__(
            title="Make Character",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.avatar = avatar
        self.locked = locked
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
            max_length=2000,
        )
        self.add_item(self.first_message)

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.description.value[-1] != ".":
            description = self.description.value + "."
        else:
            description = self.description.value
        prompt = "Your name is " + self.name.value + ". " + description + " To do an action, you surround actions in stars *like this*. You surround your dialogue in quotes " + '"like this"' + ". The person you are talking to may do the same with stars and quotes."
        response = prompt
        cut_value = 1900
        if len(response) > cut_value:
            response = response[:cut_value] + "(cont.)"
        else:
            response = response
        if self.avatar:
            avatar_image = await self.avatar.read()
            try:
                root = await interaction.send(
                    response + "\n**Do not delete this message or the character will stop working**",
                    file=discord.File(fp=io.BytesIO(avatar_image), filename="avatar.png"),
                    view=RootMessageActionsLocked() if self.locked else RootMessageActionsUnlocked())
            except Exception as e:
                await interaction.send("Failed: " + str(repr(e)))
                return
        else:
            root = await interaction.send(
                response + "\n**Do not delete this message or the character will stop working**",
                view=RootMessageActionsLocked() if self.locked else RootMessageActionsUnlocked())
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
            await thread.send(self.environment.value, view=EditEnvironmentButton())
            locked_id = interaction.user.id if self.locked else 0
            if self.avatar:
                config = MawCharacterConfig(prompt, self.environment.value, thread.id,
                                            "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/ids.txt",
                                            "./characters/" + str(root.guild.id) + "/" + str(
                                                thread.id) + "/history.txt", self.name.value, root.attachments[0].url,
                                            locked_id=locked_id, original_user_id=interaction.user.id)
            else:
                config = MawCharacterConfig(prompt, self.environment.value, thread.id,
                                            "./characters/" + str(root.guild.id) + "/" + str(thread.id) + "/ids.txt",
                                            "./characters/" + str(root.guild.id) + "/" + str(
                                                thread.id) + "/history.txt", self.name.value, None, locked_id=locked_id,
                                            original_user_id=interaction.user.id)
            make_maw_character("./characters/" + str(root.guild.id) + "/" + str(thread.id), config)
            character = MawCharacter(self.name.value, config, False)
            history = None
            if self.first_message.value != "":
                webhook = await get_webhook(root.channel)
                if self.avatar:
                    hook_message = await webhook.send(content=self.first_message.value, username=self.name.value,
                                                      avatar_url=root.attachments[0].url, wait=True, thread=thread,
                                                      view=EditMessageButton())
                else:
                    hook_message = await webhook.send(content=self.first_message.value, username=self.name.value,
                                                      wait=True, thread=thread, view=EditMessageButton())
                history = [MawCharacterMessage(self.first_message.value, hook_message.id, "character")]
            if history:
                character.write_history(history)


# this class is only used by characters
class EditMessageModal(discord.ui.Modal):
    def __init__(self, original_content, character):
        super().__init__(
            title="Edit Message",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.original_content = original_content
        self.character = character
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
        if this_message:  #do not be destructive if stuff is weird
            history[this_message[0]] = MawCharacterMessage(self.content.value, interaction.message.id, "character")
            self.character.write_history(history)
            if this_message[0] == len(
                    history) - 1 and not idx == 0:  # if this is the latest message but not the first message, add a redo button
                view = EditAndRedoMessageButton()
            else:
                view = EditMessageButton()
            await interaction.response.edit_message(content=self.content.value, view=view)


# this class is only used by characters
class EditEnvironmentModal(discord.ui.Modal):
    def __init__(self, current_prompt, config, thread):
        super().__init__(title="Edit environment", timeout=60 * 60 * 24)
        self.new_environment = discord.ui.TextInput(
            label="Environment",
            style=discord.TextInputStyle.paragraph,
            placeholder="New environment",
            default_value=current_prompt,
            required=True,
            min_length=1,
            max_length=2000
        )
        self.add_item(self.new_environment)
        self.config = config
        self.thread = thread

    async def callback(self, interaction: discord.Interaction) -> None:
        config = self.config
        config.environment_prompt = self.new_environment.value
        make_maw_character("./characters/" + str(interaction.guild.id) + "/" + str(self.thread.id), config)
        await interaction.response.edit_message(content=self.new_environment.value)


# this class is only used by characters
class EditSystemPromptModal(discord.ui.Modal):
    def __init__(self, current_prompt, config, thread):
        super().__init__(title="Edit system prompt", timeout=60 * 60 * 24)
        self.new_prompt = discord.ui.TextInput(
            label="Prompt",
            style=discord.TextInputStyle.paragraph,
            placeholder="New system prompt",
            default_value=current_prompt,
            required=True,
            min_length=30,
            max_length=4000,  # cant be above 4000
        )
        self.add_item(self.new_prompt)
        self.config = config
        self.thread = thread

    async def callback(self, interaction: discord.Interaction) -> None:
        config = self.config
        config.system_prompt = self.new_prompt.value
        make_maw_character("./characters/" + str(interaction.guild.id) + "/" + str(self.thread.id), config)
        await interaction.response.edit_message(content=str(self.new_prompt.value)[
                                                        :1900] + "\n**Do not delete this message or the character will stop working**")


# this class is only used by characters
class EditEnvironmentButton(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=None)

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary, custom_id="edit-environment")
    async def edit_environment(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.channel.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.locked_id != 0 and interaction.user.id != config.locked_id:
                await interaction.response.pong()
            else:
                await interaction.response.send_modal(
                    EditEnvironmentModal(config.environment_prompt, config, interaction.channel))


# this class is only used by character root messages
class RootMessageActionsLocked(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Edit Prompt", style=discord.ButtonStyle.primary, custom_id="edit-prompt-locked")
    async def edit_prompt(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.locked_id != 0 and interaction.user.id != config.locked_id:
                await interaction.response.pong()
            else:
                await interaction.response.send_modal(
                    EditSystemPromptModal(config.system_prompt, config, interaction.message.thread))

    @discord.ui.button(label="Unlock", style=discord.ButtonStyle.primary, custom_id="unlock")
    async def unlock_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.locked_id != 0 and interaction.user.id != config.locked_id:
                await interaction.response.pong()
            else:
                config.locked_id = 0
                make_maw_character(
                    "./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id), config)
                await interaction.response.edit_message(view=RootMessageActionsUnlocked())

    @discord.ui.button(label="Avatar", style=discord.ButtonStyle.primary, custom_id="avatar-locked")
    async def avatar_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.original_user_id != 0 and interaction.user.id != config.original_user_id:
                await interaction.response.pong()
            else:
                await interaction.response.send_message("Send a message in this channel with the new avatar.",
                                                        ephemeral=True)
                global watched_avatars
                watched_avatars.append(interaction)


# this class is only used by character root messages
class RootMessageActionsUnlocked(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Edit Prompt", style=discord.ButtonStyle.primary, custom_id="edit-prompt-unlocked")
    async def edit_prompt(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
            await interaction.response.send_modal(
                EditSystemPromptModal(config.system_prompt, config, interaction.message.thread))
        except Exception as e:
            print(repr(e))
            await interaction.response.pong()

    @discord.ui.button(label="Lock", style=discord.ButtonStyle.primary, custom_id="lock")
    async def lock_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.original_user_id != 0 and interaction.user.id != config.original_user_id:
                await interaction.response.pong()
            else:
                config.locked_id = config.original_user_id
                make_maw_character(
                    "./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id), config)
                await interaction.response.edit_message(view=RootMessageActionsLocked())

    @discord.ui.button(label="Avatar", style=discord.ButtonStyle.primary, custom_id="avatar-unlocked")
    async def avatar_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        try:
            config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.thread.id))
        except Exception as e:  # something is wrong somewhere, possibly the thread was deleted
            print(repr(e))
            await interaction.response.pong()
        else:
            if config.original_user_id != 0 and interaction.user.id != config.original_user_id:
                await interaction.response.pong()
            else:
                await interaction.response.send_message("Send a message in this channel with the new avatar.",
                                                        ephemeral=True)
                global watched_avatars
                watched_avatars.append(interaction)


# this class is only used by maw
class RedoMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary)
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.edit_message(content="...")
        if isinstance(interaction.channel, discord.DMChannel):
            config = read_config("./servers/" + str(interaction.channel.id))
        else:
            config = read_config("./servers/" + str(interaction.guild.id))
        character = MawCharacter("Maw", config, True)
        history = character.read_history()
        try:
            character.write_history(history[:-1])
        except:
            pass
        model_queue.append(
            CharacterGen(character_message=interaction.message, character=character, thread=interaction.channel,
                         user_message=None))


# this class is only used by characters
class EditAndRedoMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary, custom_id="edit-edit-and-redo-message")
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            await interaction.response.send_modal(EditMessageModal(interaction.message.content, character))

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary, custom_id="redo-edit-and-redo-message")
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        # redo button should NOT be persistent until a check for the last message is added
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            await interaction.response.edit_message(content="...")
            history = character.read_history()
            try:
                character.write_history(history[:-1])
            except:
                pass
            model_queue.append(
                CharacterGen(character_message=interaction.message, character=character, thread=interaction.channel,
                             user_message=None))

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.primary, custom_id="delete-edit-and-redo-message")
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        #Only characters use this class so delete is here
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = character.read_history()
            message_idx = None
            for idx, message in enumerate(history):
                if int(message.message_id) == interaction.message.id:
                    message_idx = idx
            if message_idx != None:
                history.pop(idx)
            character.write_history(history)
            await interaction.response.pong()
            try:
                await interaction.message.delete()
            except:
                # still kinda doesn't work but like I try
                hook = hook_list[interaction.channel.parent.id]
                await hook.delete_message(interaction.message.id)


# this class is only used by characters
class EditMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary, custom_id="edit-edit-message")
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            await interaction.response.send_modal(EditMessageModal(interaction.message.content))

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.primary, custom_id="delete-edit-message")
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        #Only characters use this class so delete is here
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = character.read_history()
            message_idx = None
            for idx, message in enumerate(history):
                if int(message.message_id) == interaction.message.id:
                    message_idx = idx
            if message_idx != None:
                history.pop(idx)
            character.write_history(history)
            await interaction.response.pong()
            try:
                await interaction.message.delete()
            except:
                # still kinda doesn't work but like I try
                hook = hook_list[interaction.channel.parent.id]
                await hook.delete_message(interaction.message.id)


# this class is only used by maw
class ResetContextButton(discord.ui.View):
    def __init__(self, *, timeout=None, history_path, ids_path):
        super().__init__(timeout=timeout)
        self.history_path = history_path
        self.ids_path = ids_path

    @discord.ui.button(label="Yes", style=discord.ButtonStyle.red)
    async def reset_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if os.path.isfile(self.history_path) and os.path.isfile(self.ids_path):
            await interaction.response.edit_message(content="Context deleted.", view=None)
            os.remove(self.history_path)
            os.remove(self.ids_path)
        else:
            await interaction.response.edit_message(content="No context found to delete.", view=None)


class MawCharacterMessage:
    def __init__(self, content, message_id, role):
        self.content = content
        self.message_id = message_id
        self.role = role


class MawCharacterConfig:
    def __init__(self, system_prompt, environment_prompt, thread_id, ids_path, history_path, name, avatar, locked_id,
                 original_user_id):
        self.system_prompt = system_prompt
        self.environment_prompt = environment_prompt
        self.thread_id = thread_id
        self.ids_path = ids_path
        self.history_path = history_path
        self.name = name
        self.avatar = avatar
        self.locked_id = locked_id
        self.original_user_id = original_user_id


class MawCharacter:
    def __init__(self, name, config, maw):
        self.ids_path = config.ids_path
        self.history_path = config.history_path
        self.name = name
        self.config = config
        self.maw = maw  # Is this maw or a character

    def write_history(self, history):
        with open(self.history_path, "w") as history_file:
            with open(self.ids_path, "w") as ids_file:
                for message in history:
                    history_file.write(message.content.replace("\n", r"\\n") + "\n")
                    role_prefix = "u" if message.role == "user" else "c"
                    ids_file.write(role_prefix + str(message.message_id) + "\n")

    def read_history(self):
        history = []
        if os.path.isfile(self.history_path):
            with open(self.history_path, "r") as history_file:
                with open(self.ids_path, "r") as ids_file:
                    history_lines, ids = history_file.readlines(), ids_file.readlines()
                    for idx, message in enumerate(history_lines):
                        role = "user" if ids[idx][:1] == "u" else "character"
                        message_id = ids[idx][1:-1]  # For simplicity with maw vs character redo, this is a string
                        history.append(MawCharacterMessage(message[:-1].replace(r"\\n", "\n"), message_id, role))
        return history


class CharacterGen:
    def __init__(self, character_message, character, thread, user_message):
        self.character_message = character_message
        self.character = character
        self.thread = thread
        self.user_message = user_message


def make_maw_character(path, config):
    os.makedirs(path, exist_ok=True)
    with open(path + "/config.txt", "w") as config_file:
        if config.thread_id:
            config_file.write(str(config.thread_id).replace("\n", r"\\n") + "\n")
        else:
            config_file.write("0\n")
        config_file.write(str(config.system_prompt.replace("\n", r"\\n")) + "\n")
        config_file.write(str(config.environment_prompt.replace("\n", r"\\n")) + "\n")
        config_file.write(str(config.name.replace("\n", r"\\n")) + "\n")
        if config.avatar: config_file.write(str(config.avatar) + "\n")
    print(config.locked_id)
    print(config.original_user_id)
    with open(path + "/locked.txt", "w") as locked_id:
        locked_id.write(str(config.locked_id))
    with open(path + "/original.txt", "w") as original_id:
        original_id.write(str(config.original_user_id))


def read_config(path):
    with open(path + "/config.txt", "r") as config_file:
        lines = [x[:-1] if x[-1] == "\n" else x for x in config_file.readlines()]  # remove new lines
    try:
        with open(path + "/locked.txt", "r") as locked_id:
            locked_id = int(locked_id.readlines()[0])
    except:
        with open(path + "/locked.txt", "w") as locked_id:
            locked_id.write("0")
        locked_id = 0
    try:
        with open(path + "/original.txt", "r") as original_id:
            original_id = int(original_id.readlines()[0])
    except:
        with open(path + "/original.txt", "w") as original_id:
            original_id.write("0")
        original_id = 0
    if len(lines) > 4:
        return MawCharacterConfig(lines[1].replace(r"\\n", "\n"), lines[2].replace(r"\\n", "\n"), int(lines[0]),
                                  path + "/ids.txt", path + "/history.txt", lines[3], lines[4], locked_id, original_id)
    else:
        return MawCharacterConfig(lines[1].replace(r"\\n", "\n"), lines[2].replace(r"\\n", "\n"), int(lines[0]),
                                  path + "/ids.txt", path + "/history.txt", lines[3], None, locked_id, original_id)


def history_to_llama(history, tokenizer, config):
    llama = []
    token_length = 0
    system_prompt = tokenizer.apply_chat_template(conversation=[{"role": "system", "content": config.system_prompt}],
                                                  tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    for idx, message in enumerate(history):
        role = "assistant" if message.role == "character" else message.role
        llama_message = [{"role": role, "content": message.content}]
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt',
                                                      add_generation_prompt=True if idx == 0 else False)
        if token_length + llama_message.shape[1] < (6500 - system_prompt.shape[1]):
            llama.append(llama_message)
            token_length += llama_message.shape[1]
        else:
            break
    if config.environment_prompt != "":
        environment_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": config.environment_prompt}], tokenize=True, return_tensors='pt',
            add_generation_prompt=False)
        if token_length + llama_message.shape[1] < (6500 - system_prompt.shape[1]):
            llama.append(environment_prompt)
            token_length += llama_message.shape[1]
    llama.append(system_prompt)
    llama.reverse()
    history.reverse()  # this was inplace so it needs to be flipped back
    llama = torch.cat(llama, 1)
    print(token_length, llama.shape)
    return llama


async def edit_add_redobutton(message, content):
    #views cannot be crafted outside of an event loop
    await message.edit(content, view=RedoMessageButton())


async def edit_add_hookredobutton(hook, message, content, thread):
    #views cannot be crafted outside of an event loop
    await hook.edit_message(content=content, message_id=message.id, thread=thread, view=EditAndRedoMessageButton())


async def edit_add_hookeditbutton(hook, message, content, thread):
    #views cannot be crafted outside of an event loop
    await hook.edit_message(content=content, message_id=message.id, thread=thread, view=EditMessageButton())


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


def message_updater(message, streamer, character, thread, channel):
    full_text = ""
    limiter = time.time()
    for text in streamer:
        print(text, flush=True, end='')
        full_text = full_text + text
        if character.maw and not isinstance(channel, discord.DMChannel):
            images = re.findall(r"<-[\S\s]+>", full_text)
            if images != None:
                with open("../DanteMode/queue.txt", "a") as image_queue:
                    for image in images:
                        full_text = full_text.replace(image, "")
                        if image[-2] == "-":
                            image = image[2:-2]
                        else:
                            image = image[2:-1]
                        image_queue.write("\n" + str(channel.id) + "|" + image[2:-1].replace("\n", "\\n"))
            pings = re.findall(r"<+[\S\s]+>", full_text)
            if pings != None:
                for ping in pings:
                    ping = ping.lower().strip()[2:-1]
                    new_ping = "No ping found. (" + ping + ")"
                    ping_cutoff = 2
                    try:
                        int(ping)
                    except:
                        for member in channel.members:
                            if len(ping) > ping_cutoff and ping in member.nick.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                            elif ping == member.nick.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                            elif len(ping) > ping_cutoff and ping in member.global_name.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                            elif ping == member.global_name.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                            elif len(ping) > ping_cutoff and ping in member.name.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                            elif ping == member.name.lower().strip():
                                new_ping = "<@" + str(member.id) + ">"
                    else:
                        if int(ping) in [x.id for x in channel.members]:
                            new_ping = ping
                        else:
                            for member in channel.members:
                                if len(ping) > ping_cutoff and ping in member.nick.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                                elif ping == member.nick.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                                elif len(ping) > ping_cutoff and ping in member.global_name.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                                elif ping == member.globalname.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                                elif len(ping) > ping_cutoff and ping in member.name.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                                elif ping == member.name.lower().strip():
                                    new_ping = "<@" + str(member.id) + ">"
                    full_text = full_text.replace(ping, new_ping)
        if time.time() - limiter > 0.8:
            limiter = time.time()
            if character.maw:
                asyncio.run_coroutine_threadsafe(coro=message.edit(full_text), loop=client.loop)
            else:
                asyncio.run_coroutine_threadsafe(coro=temp_edit(message.id, thread, full_text, channel.id),
                                                 loop=client.loop)
    global model_queue
    if character.maw:
        if not message.channel.id in [x.character_message.channel.id for x in model_queue[1:]]:
            asyncio.run_coroutine_threadsafe(coro=edit_add_redobutton(message, full_text),
                                             loop=client.loop)
        else:
            asyncio.run_coroutine_threadsafe(coro=message.edit(full_text), loop=client.loop)
    else:
        if not thread.id in [x.character_message.channel.id for x in model_queue[1:]]:
            asyncio.run_coroutine_threadsafe(
                coro=edit_add_hookredobutton(hook_list[channel.id], message, full_text,
                                             thread), loop=client.loop)
        else:
            asyncio.run_coroutine_threadsafe(
                coro=edit_add_hookeditbutton(hook_list[channel.id], message, full_text,
                                             thread), loop=client.loop)


async def async_watcher():
    global all_tokens
    global all_time
    global model_queue
    model = None
    tokenizer = AutoTokenizer.from_pretrained(
        #"failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )  # can just be kept loaded
    stop_token = tokenizer.encode("<|eot_id|>")
    while True:
        if model_queue == []:
            if model != None:
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                vram.deallocate("Maw")
            time.sleep(0.01)
        else:
            if all_tokens != 0:
                asyncio.run_coroutine_threadsafe(coro=client.change_presence(
                    activity=discord.Activity(type=discord.ActivityType.watching,
                                              name="at " + str(round(all_tokens / all_time, 2)) + " avg tps"),
                    status=discord.Status.online), loop=client.loop)
            current_gen = model_queue[0]
            if model == None:
                print("allocating memory")
                vram.allocate("Maw")
                print("request sent")
                async for i in vram.wait_for_allocation("Maw"):
                    if current_gen.character.maw:
                        asyncio.run_coroutine_threadsafe(coro=current_gen.character_message.edit(
                            "(Waiting for " + str(i) + " before loading model.)"), loop=client.loop)
                print("memory allocated, loading model")
                model = AutoModelForCausalLM.from_pretrained(
                    "failspy/Meta-Llama-3-8B-Instruct-abliterated-v3",
                    #"cogvlm-abliterated",
                    local_files_only=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                    load_in_8bit=False,
                )
            gc.collect()
            torch.cuda.empty_cache()
            history = current_gen.character.read_history()
            if current_gen.user_message != None:
                history.append(current_gen.user_message)
                current_gen.character.write_history(
                    history)  # if message is edited or deleted during generation, it needs to be reflected
            model_input = history_to_llama(history, tokenizer, current_gen.character.config)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            if isinstance(current_gen.thread, discord.Thread):
                thread, channel = current_gen.thread, current_gen.thread.parent
            else:
                thread, channel = None, current_gen.thread
            streamer_thread = threading.Thread(target=message_updater,
                                               args=[current_gen.character_message, streamer, current_gen.character,
                                                     thread, channel])
            streamer_thread.start()
            start_time = time.time()
            response = model.generate(input_ids=model_input.to('cuda'), **model_args, streamer=streamer,
                                      eos_token_id=stop_token)
            all_tokens += len(response[0][model_input.shape[1]:])
            all_time += time.time() - start_time
            asyncio.run_coroutine_threadsafe(coro=client.change_presence(
                activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(
                    round(len(response[0][model_input.shape[1]:]) / (time.time() - start_time), 2)) + " tps | " + str(
                    round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.idle), loop=client.loop)
            decoded_response = tokenizer.decode(response[0][model_input.shape[1]:], skip_special_tokens=True)
            if current_gen.character.maw:
                history.append(MawCharacterMessage(decoded_response, (str(current_gen.character_message.id) + "-" + str(
                    current_gen.character_message.channel.id)), "character"))
            else:
                history.append(MawCharacterMessage(decoded_response, current_gen.character_message.id, "character"))
            current_gen.character.write_history(history)
            del response, decoded_response, model_input
            gc.collect()
            torch.cuda.empty_cache()
            model_queue.pop(0)


def watcher():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_watcher())


@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(status=discord.Status.idle)
    client.add_view(RootMessageActionsUnlocked())
    client.add_view(RootMessageActionsLocked())
    client.add_view(EditEnvironmentButton())
    client.add_view(EditAndRedoMessageButton())
    client.add_view(EditMessageButton())


@client.event
async def on_message(message):
    global model_queue
    global second_last_message
    global last_message
    global watched_avatars
    maw_response = False
    character_response = False
    dm = False
    if isinstance(message.channel, discord.TextChannel):
        if "maw," in message.content.lower() and not r"\end" in message.content.lower() and not "/end" in message.content.lower(): maw_response = True
        try:
            if last_message[message.channel.id].author.id == client.user.id and second_last_message[
                message.channel.id].author.id == message.author.id and not message.author.bot and not r"\end" in message.content and not "/end" in message.content:
                maw_response = True
        except:
            pass
    if isinstance(message.channel, discord.Thread):
        if os.path.isdir("./characters/" + str(message.guild.id) + "/" + str(message.channel.id)):
            character_response = True
            maw_response = False
        elif "maw," in message.content.lower() and not r"\end" in message.content.lower() and not "/end" in message.content.lower():
            maw_response = True
        else:
            try:
                if last_message[message.channel.id].author.id == client.user.id and second_last_message[
                    message.channel.id].author.id == message.author.id and not message.author.bot and not r"\end" in message.content and not "/end" in message.content:
                    maw_response = True
            except:
                pass
    if isinstance(message.channel, discord.DMChannel):
        maw_response = True
        dm = True
    if message.author.bot:
        character_response = False
        maw_response = False
    try:
        second_last_message[message.channel.id] = last_message[message.channel.id]
    except:
        pass
    last_message[message.channel.id] = message
    if message.attachments:
        if message.author.id in [x.user.id for x in watched_avatars]:
            index = [x.user.id for x in watched_avatars].index(message.author.id)
            avatar_interaction = watched_avatars[index]
            if avatar_interaction.channel == message.channel:
                new_avatar = message.attachments[0]
                if not new_avatar.content_type == "image/jpeg" and not new_avatar.content_type == "image/png":
                    await message.channel.send("Avatar is not png or jpg. Press the button to try again.")
                else:
                    new_avatar = await new_avatar.to_file()
                    try:
                        await message.delete()
                    except:
                        pass
                    try:
                        new_avatar = await avatar_interaction.message.edit(file=new_avatar)
                    except:
                        await message.channel.send("Failed to set new avatar!")
                    else:
                        try:
                            config = read_config("./characters/" + str(message.guild.id) + "/" + str(avatar_interaction.message.thread.id))
                            config.avatar = new_avatar.attachments[0].url
                            make_maw_character("./characters/" + str(message.guild.id) + "/" + str(avatar_interaction.message.thread.id),
                                               config)
                            await message.channel.send("New avatar set")
                        except:
                            await message.channel.send("Failed to set new avatar! Thread may have been deleted.")
                watched_avatars.pop(index)
    if maw_response:
        maw_message = await message.channel.send("...")
        old_message_id = None
        relative_path = "./servers/" + str(message.channel.id) if dm else "./servers/" + str(message.guild.id)
        if os.path.isdir(relative_path):
            config = read_config(relative_path)
            character = MawCharacter("Maw", config, True)
            if os.path.isfile(relative_path + "/history.txt"):
                history = character.read_history()
                try:
                    old_message_id = (
                        int(history[-1].message_id.split("-")[-2]), int(history[-1].message_id.split("-")[-1]))
                except:
                    pass
        else:
            if isinstance(message.channel, discord.DMChannel):
                system_prompt = "You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. You are talking to " + (
                    message.author.global_name if message.author.global_name else message.author.name)
            else:
                system_prompt = "You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. The name of the user you are talking to is included in the message. You can also make images. To do so, enclose a description of the image in '<-' and '>', like this: <-prompt>. Do not make ASCII art or just describe the image without enclosing it unless specifically stated, and remember to always start image prompts with <- and end them with >. To ping users, enclose either their name or ID in '<+' and '>', like this: <+Edna>."
            config = MawCharacterConfig(system_prompt, "", None, relative_path + "/ids.txt",
                                        relative_path + "/history.txt", "Maw", None, 0, 0)
            make_maw_character(relative_path, config)
            character = MawCharacter("Maw", config, True)
        #history = character.read_history()
        #history.append(MawCharacterMessage(message.content, str(message.id), "user"))
        #character.write_history(history)  # if message is edited or deleted during generation, it needs to be reflected
        user_message = MawCharacterMessage(content=(
                                                       message.author.global_name if message.author.global_name else message.author.name) + " said: " + message.content.strip(),
                                           message_id=str(message.id), role="user")
        model_queue.append(CharacterGen(character_message=maw_message, character=character, thread=message.channel,
                                        user_message=user_message))
        try:
            if isinstance(message.channel, discord.DMChannel):
                old_message = await message.channel.fetch_message(old_message_id[0])
                await old_message.edit(old_message.content, view=None)
            else:
                if old_message_id:
                    channel = client.get_channel(old_message_id[1])
                    old_message = await channel.fetch_message(old_message_id[0])
                await old_message.edit(old_message.content, view=None)
        except:
            pass
    if character_response:
        hook = await get_webhook(message.channel.parent)
        config = read_config("./characters/" + str(message.guild.id) + "/" + str(message.channel.id))
        if config.locked_id == 0 or message.author.id == config.locked_id:
            if config.avatar:
                character_message = await hook.send(content="...", username=config.name, wait=True,
                                                    thread=message.channel, avatar_url=config.avatar)
            else:
                character_message = await hook.send(content="...", username=config.name, wait=True,
                                                    thread=message.channel)
            character = MawCharacter(config.name, config, False)
            old_message_id = None
            if os.path.isfile("./characters/" + str(message.guild.id) + "/" + str(message.channel.id) + "/history.txt"):
                history = character.read_history()
                try:
                    old_message_id = int(history[-1].message_id)
                except:
                    pass
            #history = character.read_history()
            #history.append(MawCharacterMessage(message.content, str(message.id), "user"))
            #character.write_history(history) # if message is edited or deleted during generation, it needs to be reflected
            user_message = MawCharacterMessage(content=message.content, message_id=str(message.id), role="user")
            model_queue.append(
                CharacterGen(character_message=character_message, character=character, thread=message.channel,
                             user_message=user_message))
            if old_message_id:
                try:
                    await hook.edit_message(message_id=old_message_id, view=EditMessageButton(), thread=message.channel)
                except:
                    pass  # isn't really needed, but I don't like random error messages in my console


@client.event
async def on_raw_message_edit(payload):
    channel = client.get_channel(payload.channel_id)
    try:
        payload.data["content"]
    except:
        pass
    else:
        if isinstance(channel, discord.Thread) and os.path.exists(
                "./characters/" + str(channel.guild.id) + "/" + str(channel.id) + "/"):
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
                    history[message_idx] = MawCharacterMessage(payload.data["content"], history[message_idx].message_id,
                                                               history[message_idx].role)
                character.write_history(history)


@client.event
async def on_raw_message_delete(payload):
    channel = client.get_channel(payload.channel_id)
    if isinstance(channel, discord.Thread) and os.path.exists(
            "./characters/" + str(channel.guild.id) + "/" + str(channel.id) + "/"):
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
            if edit_message.role == "character":
                edit_message = await channel.fetch_message(edit_message.message_id)
                hook = await get_webhook(channel.parent)
                await hook.edit_message(message_id=edit_message.id, thread=channel, view=EditAndRedoMessageButton())


@client.slash_command(description="Sends a form to make a character", dm_permission=False)
async def character(
        interaction: discord.Interaction,
        avatar: Optional[discord.Attachment] = discord.SlashOption(
            name="avatar",
            required=False,
            description="An avatar for your character. Must be jpg or png",
        ),
        locked: Optional[bool] = discord.SlashOption(
            name="locked",
            required=False,
            description="Locks a session to only the current user.",
        ),
):
    if avatar and not avatar.content_type == "image/jpeg" and not avatar.content_type == "image/png":
        await interaction.response.send_message("Avatar is not png or jpg. Please try again")
    else:
        modal = CharacterModal(avatar, locked)
        await interaction.response.send_modal(modal)


@client.slash_command(description="Resets the context of Maw for the whole server (not including characters)")
async def reset(
        interaction: discord.Interaction,
):
    if interaction.guild != None:
        relative_path = "./servers/" + str(interaction.guild.id)
    else:
        relative_path = "./servers/" + str(interaction.channel_id)
    if os.path.isfile(relative_path + "/history.txt") and os.path.isfile(relative_path + "/ids.txt"):
        await interaction.response.send_message(
            "Are you sure? This will delete Maws memory in this server, not including characters.",
            view=ResetContextButton(history_path=relative_path + "/history.txt", ids_path=relative_path + "/ids.txt"))
    else:
        await interaction.response.send_message("No context found to clear.")


threading.Thread(target=watcher).start()
client.run(TOKEN)
