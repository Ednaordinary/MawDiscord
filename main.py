import array
import io
import itertools
import json
import os
import gc
import re
import sys
import wave
from queue import Queue
from struct import unpack_from

import librosa
import nextcord as discord
import numpy as np
import soundfile

import decrypter
from recorder import VoiceRecvClient, BytesSRAudioSource
from dotenv import load_dotenv
from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
import time
import torch
import threading
import asyncio
from typing import Optional
import vram
from random import randint
import speech_recognition as sr
import datetime
import whisper
from pyogg import OpusDecoder
from scipy.signal import resample
from scipy.io import wavfile

model_queue = []
hook_list = {}  # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.
second_last_message = {}
last_message = {}
watched_avatars = []
voice_data = {}
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.all()
client = discord.AutoShardedClient(intents=intents)
discord.opus._load_default()
all_tokens = 0
all_time = 0
whisper_model = None
whisperloading = False
whisperusers = 0
model_callback_limiter = 1 # doesn't matter what this value is
maw_voice_channels = []
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


dm_system_prompt = "You are Maw, an intelligence model that answers questions to the best of your knowledge. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. You are talking to "
regular_system_prompt = "\nYou are Maw, an intelligence model that answers questions to the best of your knowledge. Depending on the questions you are asked, you can either go into detail or be brief with your explanations. If the user asks a coding question, use standard markdown formatting for code. Do not refuse to answer questions. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. The name of the user you are talking to is included in the message.\n\nYou are able to generate or create images by making a generation prompt: enclose a description of an image in <- and ->, for example: <-A hot dog on a grill, the grill sits on a wooden table with condiments on it->. Do not generate an image unless explicitly asked to do so.\nIf asked to generate an image, add a short description or acknowledgement before you add the generation prompt.\n\nYou are talking in a server named "


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

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary, custom_id="maw-redo")
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if isinstance(interaction.channel, discord.DMChannel):
            config = read_config("./servers/" + str(interaction.channel.id))
        else:
            config = read_config("./servers/" + str(interaction.guild.id))
        character = MawCharacter("Maw", config, True)
        history = character.read_history()
        if int(history[-1].message_id.split("-")[0]) == interaction.message.id:
            try:
                character.write_history(history[:-1])
            except:
                pass
            await interaction.response.edit_message(content="...")
            model_queue.append(
                CharacterGen(character_message=interaction.message, character=character, thread=interaction.channel,
                             user_message=None, vc=False))
        else:
            await interaction.response.edit_message(view=None)


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
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = character.read_history()
            if int(history[-1].message_id) == interaction.message.id:
                try:
                    character.write_history(history[:-1])
                except:
                    pass
                await interaction.response.edit_message(content="...")
                model_queue.append(
                    CharacterGen(character_message=interaction.message, character=character, thread=interaction.channel,
                                 user_message=None, vc=False))
            else:
                await interaction.response.edit_message(view=None)

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
            await interaction.response.send_modal(EditMessageModal(interaction.message.content, character))

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
    def __init__(self, character_message, character, thread, user_message, vc):
        self.character_message = character_message
        self.character = character
        self.thread = thread
        self.user_message = user_message
        self.vc = vc


class MawVoiceSession:
    def __init__(self, guild, message, thread, proto, exclusive):
        self.guild = guild
        self.message = message
        self.thread = thread
        self.proto = proto
        self.exclusive = exclusive

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


def history_to_llama(history, config):
    tokenizer = AutoTokenizer.from_pretrained(
        "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    )
    llama = []
    token_length = 0
    print(config.system_prompt)
    system_prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "system", "content": config.system_prompt.replace(r"\n", "\n")}],
        tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    for idx, message in enumerate(history):
        role = "assistant" if message.role == "character" else message.role
        llama_message = [{"role": role, "content": message.content.replace(r"\n", "\n")}]
        print(message.content)
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt',
                                                      add_generation_prompt=True if idx == 0 else False)
        if token_length + llama_message.shape[1] < (128000 - system_prompt.shape[1]):
            llama.append(llama_message)
            token_length += llama_message.shape[1]
        else:
            break
    if config.environment_prompt.strip() != "":
        environment_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": config.environment_prompt.replace(r"\n", "\n")}],
            tokenize=True, return_tensors='pt',
            add_generation_prompt=False)
        if token_length + environment_prompt.shape[1] < (128000 - system_prompt.shape[1]):
            llama.append(environment_prompt)
            token_length += environment_prompt.shape[1]
    llama.append(system_prompt)
    llama.reverse()
    history.reverse()  # this was inplace so it needs to be flipped back
    llama = torch.cat(llama, 1)
    print("tokens:", token_length)
    #llama = str([tokenizer.decode(x, skip_special_tokens=False) for x in llama])
    llama = tokenizer.encode(tokenizer.batch_decode(llama, skip_special_tokens=False)[0].replace(r"\n", "\n"),
                             add_special_tokens=True, return_tensors='pt')
    #llama = "".join(llama)
    #llama = llama.replace(r"\n", "\n")
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


async def get_webhook(channel) -> discord.Webhook:
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
    try:
        await hook_list[channel_id].edit_message(message_id=message_id, content=content, thread=thread_id)
    except Exception as e:
        print(repr(e))


async def async_watcher():
    global all_tokens
    global all_time
    global model_queue
    model = None
    cache = None
    model_dir = "./llama-3.1-8b-instruct-abliterated-exl2-7.5bpw"
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    tokenizer = ExLlamaV2Tokenizer(config)
    while True:
        if model_queue == []:
            if model != None:
                model = None
                cache = None
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
                if isinstance(current_gen.thread, discord.Thread):
                    thread, channel = current_gen.thread, current_gen.thread.parent
                else:
                    thread, channel = None, current_gen.thread
                model = ExLlamaV2(config)
                cache = ExLlamaV2Cache_Q8(model, lazy=True, max_seq_len=500 * 256)
                global model_callback_limiter # no concurrent load, so global is okay
                model_callback_limiter = time.time()
                def model_load_callback(current, total):
                    print(current, total)
                    global model_callback_limiter
                    if time.time() > model_callback_limiter + 0.5: # only bursts for the first second or so, so its rate can be higher
                        if current_gen.character.maw:
                            asyncio.run_coroutine_threadsafe(coro=current_gen.character_message.edit(str(int(current*100 / total)) + "%"),
                                                             loop=client.loop)
                        else:
                            asyncio.run_coroutine_threadsafe(
                                coro=temp_edit(current_gen.character_message.id, thread, str(int(current*100 / total)) + "%", channel.id),
                                loop=client.loop)
                        model_callback_limiter = time.time()
                model.load_autosplit(cache, progress=False, callback=model_load_callback)
            gc.collect()
            torch.cuda.empty_cache()
            history = current_gen.character.read_history()
            if current_gen.user_message != None:
                history.append(current_gen.user_message)
                current_gen.character.write_history(
                    history)  # if message is edited or deleted during generation, it needs to be reflected
            model_input = history_to_llama(history, current_gen.character.config)
            message = current_gen.character_message
            character = current_gen.character
            tokens = 0
            limiter = time.time()
            generator = ExLlamaV2DynamicGenerator(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                max_q_size=16,
            )
            #input_ids = tokenizer.encode(model_input, add_bos=False, encode_special_tokens=True)
            input_ids = model_input
            print(model_input)
            sampler = ExLlamaV2Sampler.Settings.greedy()
            sampler.top_p = 0.9
            sampler.temperature = 0.6
            #sampler.min_temp = 0.6
            #sampler.max_temp = 0.7
            sampler.token_repetition_penalty = randint(1150, 1250) / 1000
            sampler.token_repetition_range = 50
            sampler.token_repetition_decay = 10
            job = ExLlamaV2DynamicJob(
                input_ids=input_ids,
                max_new_tokens=768,
                token_healing=True,
                #stop_conditions="<|eot_id|>",
                gen_settings=sampler,
                decode_special_tokens=True,
                seed=randint(1, 10000000),
            )
            response = ""
            final_response = ""
            generator.enqueue(job)
            eos = False
            start_time = time.time()
            while not eos:
                results = generator.iterate()
                if results != []:
                    result = results[0]
                    if result["stage"] == "streaming":
                        text = result.get("text", "").replace(r"\n", "\n")
                        all_tokens += 1
                        tokens += 1
                        print(text, end="", flush=True)
                        response = response + text
                        final_response = final_response + text
                        if "<|eot_id|>" in response:
                            eos = True
                            final_response = final_response.replace("<|eot_id|>", "")
                            response = response.replace("<|eot_id|>", "")
                            if "<-" in response and character.maw and not isinstance(
                                    message.channel, discord.DMChannel):
                                find_image = re.compile(r'<-[\S\s]+>')
                                for image in re.findall(find_image, final_response):
                                    print("image detected")
                                    final_response = final_response.replace(image, "")
                                    image = image[2:-1]
                                    if image[-1] == "-": image = image[:-1]
                                    if image != "":
                                        with open("../DanteMode/queue.txt", "a") as image_queue:
                                            image_queue.write(
                                                "\n" + str(channel.id) + "|" + str(image).replace("\n", "\\n"))
                        if time.time() - limiter > 1.2:
                            limiter = time.time()
                            if character.maw:
                                asyncio.run_coroutine_threadsafe(coro=message.edit(final_response), loop=client.loop)
                            else:
                                asyncio.run_coroutine_threadsafe(
                                    coro=temp_edit(message.id, thread, final_response, channel.id),
                                    loop=client.loop)
                else:
                    eos = True
            if character.maw:
                if not message.channel.id in [x.character_message.channel.id for x in model_queue[1:]]:
                    asyncio.run_coroutine_threadsafe(coro=edit_add_redobutton(message, final_response),
                                                     loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=message.edit(final_response), loop=client.loop)
            else:
                if not thread.id in [x.character_message.channel.id for x in model_queue[1:]]:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_add_hookredobutton(hook_list[channel.id], message, final_response,
                                                         thread), loop=client.loop)
                    except:
                        pass
                else:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_add_hookeditbutton(hook_list[channel.id], message, final_response,
                                                         thread), loop=client.loop)
                    except:
                        pass
            all_time += time.time() - start_time
            asyncio.run_coroutine_threadsafe(coro=client.change_presence(
                activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(
                    round(tokens / (time.time() - start_time), 2)) + " tps | " + str(
                    round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.idle), loop=client.loop)
            if current_gen.character.maw:
                history.append(MawCharacterMessage(response, (str(current_gen.character_message.id) + "-" + str(
                    current_gen.character_message.channel.id)), "character"))
            else:
                history.append(MawCharacterMessage(response, current_gen.character_message.id, "character"))
            current_gen.character.write_history(history)
            del response, generator, job, input_ids, result, results, model_input
            gc.collect()
            torch.cuda.empty_cache()
            model_queue.pop(0)


def watcher():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_watcher())


class TimeStampedVoiceData:
    def __init__(self, data, time):
        self.data = data
        self.time = time

class OpusFrame:
    def __init__(
        self,
        sequence: int,
        timestamp: float,
        received_timestamp: float,
        ssrc: int,
        decrypted_data: Optional[bytes],
        decoded_data: Optional[bytes] = None,
        user_id: Optional[int] = (None),
    ):
        self.sequence = sequence
        self.timestamp = timestamp
        self.received_timestamp = received_timestamp
        self.ssrc = ssrc
        self.decrypted_data = decrypted_data
        self.decoded_data = decoded_data
        self.user_id = user_id

    @property
    def is_silent(self):
        return self.decrypted_data == b"\xf8\xff\xfe"

    def __repr__(self) -> str:
        attrs = (
            ("sequence", self.sequence),
            ("timestamp", self.timestamp),
            ("received_timestamp", self.received_timestamp),
            ("ssrc", self.ssrc),
            ("user_id", self.user_id),
        )
        joined = " ".join("%s=%r" % t for t in attrs)
        return f"<{self.__class__.__name__} {joined}>"

async def async_voice_channel_listener(proto, session):
    print("Starting channel listener")
    try:
        global voice_data
        decoder = OpusDecoder()
        decoder.set_channels(2)
        decoder.set_sampling_frequency(48000)
        async for data in proto.listen():
            if 200 <= data[1] <= 204:
                continue
            data = bytearray(data)
            header = data[:12]
            data = data[12:]
            sequence, timestamp, ssrc = unpack_from(">xxHII", header)
            decrypted = proto.decrypt(header, data)
            opus_frame = OpusFrame(sequence, timestamp, time.perf_counter(), ssrc, decrypted)
            user_id = proto._wait_for_user_id(ssrc)
            try:
                voice_data[session][user_id]
            except:

                decoded = decoder.decode(opus_frame.decrypted_data).tobytes()
                voice_data[session][user_id] = array.array('B')
                voice_data[session][user_id].extend(decoded)
            else:
                decoded = decoder.decode(opus_frame.decrypted_data).tobytes()
                voice_data[session][user_id].extend(decoded)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        raise e


def voice_channel_listener(proto, session):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_voice_channel_listener(proto, session))


def load_whisper():
    #this is done in a new thread because it HAS to complete successfully
    global whisper_model
    global whisperloading
    if not whisper_model:
        if not whisperloading:
            whisperloading = True
            try:
                whisper_model = whisper.load_model("medium.en", device='cuda')
            except Exception as e:
                print(repr(e))
                pass
            whisperloading = False
        else:
            while whisper_model == None:
                time.sleep(0.01)



def request_whisper_text(audio):
    global whisper_model
    while whisper_model == None:
        time.sleep(0.01)
    transcribed_text = whisper_model.transcribe(audio, fp16=True, hallucination_silence_threshold=2)
    return transcribed_text


def user_listener(session, user, proto):
    global voice_data
    print("Started user listener, adjusting audio")
    source = BytesSRAudioSource(voice_data[session][user.id])
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 1200
    recognizer.dynamic_energy_threshold = True
    recognizer.adjust_for_ambient_noise(source)
    dq = Queue()
    rt = 20.0
    pto = 4.0
    pt = None
    transcript = ['']
    def record_callback(_, audio:sr.AudioData) -> None:
        data = audio.get_raw_data()
        dq.put(data)
    recognizer.listen_in_background(source, record_callback, phrase_time_limit=rt)
    print("Finished setting up audio")
    # num = 0
    while proto.is_connected():
        if not dq.empty():
            now = datetime.datetime.now(datetime.UTC)
            phrase_complete = False
            if pt and now - pt > datetime.timedelta(seconds=pto):
                phrase_complete = True
            pt = now
            audio_data = b''.join(dq.queue)
            dq.queue.clear()
            #audio_np = np.array(audio_data).astype(np.float32) / (2**15)
            #resampled = resample(audio_np, num=(len(audio_data)*16000/48000))
            #resampled = resampled / 32768.0
            new_samples = round(len(audio_data) * 16000.0 / 48000)
            print(new_samples)
            if new_samples != 0: # only transcribe if there is something to transcribe
                audiobytes = io.BytesIO()
                wave_write = wave.open(audiobytes, "wb")
                wave_write.setnchannels(1)
                wave_write.setsampwidth(2)
                wave_write.setframerate(48000)
                wave_write.writeframes(audio_data)
                wave_write.close()
                audiobytes.seek(0)
                audio_np = librosa.load(audiobytes, sr=16000)[0]
                result = request_whisper_text(audio_np)
                text = result['text'].strip()
                if phrase_complete:
                    transcript.append(text)
                else:
                    transcript[-1] = text
            #if "." in transcript[-1] or "?" in transcript[-1] or "!" in transcript[-1]:
            if ''.join(transcript).strip() != '' and ''.join(transcript).strip() != 'you':
                hook = asyncio.run_coroutine_threadsafe(coro=get_webhook(session.thread.parent),
                                                 loop=client.loop).result()
                user_hook_message = asyncio.run_coroutine_threadsafe(coro=hook.send(content=''.join(transcript), username=user.global_name, avatar_url=user.display_avatar.url, thread=session.thread),
                    loop=client.loop)
                if not session.exclusive:
                    print("Session isn't exclusive")
                    if "." in ''.join(transcript) or "?" in ''.join(transcript) or "!" in ''.join(transcript):
                        print("Found prompt in text")
                        # I could simply use on_message, but it's easier, faster to handle it here
                        user_hook_message = user_hook_message.result()
                        maw_message = asyncio.run_coroutine_threadsafe(coro=session.thread.send("..."),
                            loop=client.loop)
                        relative_path = "./servers/" + str(session.guild.id)
                        if os.path.isdir(relative_path):
                            config = read_config(relative_path)
                            config.system_prompt = config.system_prompt + session.guild.name + ", connected to the voice channel " + session.proto.channel.name
                            character = MawCharacter("Maw", config, True)
                            if os.path.isfile(relative_path + "/history.txt"):
                                history = character.read_history()
                        else:
                            system_prompt = regular_system_prompt
                            config = MawCharacterConfig(system_prompt, "", None, relative_path + "/ids.txt",
                                                        relative_path + "/history.txt", "Maw", None, 0, 0)
                            make_maw_character(relative_path, config)
                            config.system_prompt = config.system_prompt + session.guild.name + ", connected to the voice channel " + session.proto.channel.name
                            character = MawCharacter("Maw", config, True)
                        user_message = MawCharacterMessage(content=user.global_name + " said: " + ''.join(transcript),
                                                           message_id=str(user_hook_message.id), role="user")
                        model_queue.append(
                            CharacterGen(character_message=maw_message, character=character, thread=session.thread,
                                         user_message=user_message, vc=session))
                transcript = ['']
                pass
        else:
            time.sleep(0.01)


def voice_channel_watcher(session):
    guild = session.guild
    message = session.message
    thread = session.thread
    proto = session.proto
    global voice_data
    voice_data[session] = {}
    print("Started watcher")
    threading.Thread(target=voice_channel_listener, args=[proto, session]).start()
    user_threads = {}
    print("Got to scalable connections state")
    while proto.is_connected():
        for user_id in voice_data[session].keys():
            if user_id not in user_threads.keys():
                print(user_id)
                user = client.get_user(user_id)
                print(user)
                user_threads[user_id] = threading.Thread(target=user_listener, args=[session, user, proto])
                user_threads[user_id].start()
        time.sleep(0.01)
    print("No longer connected")
    del voice_data[session]
    global maw_voice_channels
    for x, idx in enumerate(maw_voice_channels):
        if x == session:
            maw_voice_channels.pop(idx)
    global whisperusers
    whisperusers -= 1
    if whisperusers == 0:
        global whisper_model
        whisper_model = None
        gc.collect()
        torch.cuda.empty_cache()
        global whisperloading
        whisperloading = False


@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')
    await client.change_presence(status=discord.Status.idle)
    client.add_view(RootMessageActionsUnlocked())
    client.add_view(RootMessageActionsLocked())
    client.add_view(EditEnvironmentButton())
    client.add_view(EditAndRedoMessageButton())
    client.add_view(EditMessageButton())
    client.add_view(RedoMessageButton())


@client.event
async def on_message(message):
    global model_queue
    global second_last_message
    global last_message
    global watched_avatars
    global maw_voice_channels
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
    if message.channel in [x.thread for x in maw_voice_channels]:
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
                            config = read_config("./characters/" + str(message.guild.id) + "/" + str(
                                avatar_interaction.message.thread.id))
                            config.avatar = new_avatar.attachments[0].url
                            make_maw_character("./characters/" + str(message.guild.id) + "/" + str(
                                avatar_interaction.message.thread.id),
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
            if not isinstance(message.channel, discord.DMChannel):
                config.system_prompt = config.system_prompt + message.guild.name + " in channel " + message.channel.name
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
                system_prompt = dm_system_prompt + (
                    message.author.global_name if message.author.global_name else message.author.name)
            else:
                system_prompt = regular_system_prompt
            config = MawCharacterConfig(system_prompt, "", None, relative_path + "/ids.txt",
                                        relative_path + "/history.txt", "Maw", None, 0, 0)
            make_maw_character(relative_path, config)
            if not isinstance(message.channel, discord.DMChannel):
                config.system_prompt = config.system_prompt + message.guild.name + " in channel " + message.channel.name
            character = MawCharacter("Maw", config, True)
        #history = character.read_history()
        #history.append(MawCharacterMessage(message.content, str(message.id), "user"))
        #character.write_history(history)  # if message is edited or deleted during generation, it needs to be reflected
        user_message = MawCharacterMessage(content=(
                                                       message.author.global_name if message.author.global_name else message.author.name) + " said: " + message.content.strip(),
                                           message_id=str(message.id), role="user")
        model_queue.append(CharacterGen(character_message=maw_message, character=character, thread=message.channel,
                                        user_message=user_message, vc=False))
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
                             user_message=user_message, vc=False))
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

@client.slash_command(description="Maw for voice channels", dm_permission=False)
async def voice(
        interaction: discord.Interaction,
        channel: discord.VoiceChannel,
        transcribe_only: Optional[bool] = discord.SlashOption(
            name="transcribe_only",
            required=False,
            description="Whether or not Maw responds to what you ask.",
        ),
):
    global maw_voice_channels
    if not transcribe_only: transcribe_only = False
    if not interaction.guild in [x.guild for x in maw_voice_channels]:
        try:
            proto = await channel.connect(cls=VoiceRecvClient)
            sent_message = await interaction.response.send_message("Starting a voice session in " + str(channel.name))
            sent_message = await sent_message.fetch()
            thread = await sent_message.create_thread(name="Maw Voice Session")
            session = MawVoiceSession(guild=interaction.guild, message=sent_message, thread=thread, proto=proto, exclusive=transcribe_only)
            maw_voice_channels.append(session)
            load_whisper()
            global whisperusers
            whisperusers += 1
            threading.Thread(target=voice_channel_watcher, args=[session]).start()
        except Exception as e:
            print(repr(e))
            await interaction.response.send_message("Failed to connect to that channel!")
    else:
        await interaction.response.send_message("Already connected to a channel in this server!")

threading.Thread(target=watcher).start()
client.run(TOKEN)
