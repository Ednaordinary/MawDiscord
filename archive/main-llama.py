import array
import io
import os
import gc
import re
import sys
import wave
from queue import Queue
from struct import unpack_from
import math
import librosa
import nextcord as discord
import numpy as np
import scipy
from recorder import VoiceRecvClient, BytesSRAudioSource
from dotenv import load_dotenv
from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Cache_Q8, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler, \
    ExLlamaV2DynamicJobAsync, ExLlamaV2DynamicGeneratorAsync
import time
import torch
import torchaudio
import threading
import asyncio
from typing import Optional
import vram
from random import randint
import random
import speech_recognition as sr
import datetime
import whisper
import outetts
from pyogg import OpusDecoder
from scipy.io import wavfile
#from numba import cuda as numba_cuda

model = None
modelusers = []
modeltimeout = None
generator = None
handled_requests = []
model_queue = []
hook_list = {}  # Hooks must be renewed every bot launch otherwise we can't add buttons to webhook messages.
second_last_message = {}
last_message = {}
watched_avatars = []
voice_data = {}
voice_queue = {}
voice_play = {}
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
whispertranscribing = False
speech_model = None
speechloading = None
watched_channels = []
stay_allocated = 0
typing_channels = []
exclusive = {}
model_callback_limiter = 1  # doesn't matter what this value is
maw_voice_channels = []
maw_auto_servers = []  # EXPERIMENTAL
maw_auto_watched = {}
maw_auto_sample = {}
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#numba_device = numba_cuda.get_current_device()
#numba_device.reset()

dm_system_prompt = "You are Maw, an empathetic intelligence model that answers questions to the best of your knowledge. Depending on the questions you are asked, you can either go into detail or be brief with your explanations. If the user asks a coding question, use standard markdown formatting for code. Do not refuse to answer questions. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. The name of the user you are talking to is included in each message. You are also capable of emulating emotions and opinions about topics. You are talking to "
regular_system_prompt = "\nYou are Maw, an empathetic intelligence model that answers questions to the best of your knowledge. Depending on the questions you are asked, you can either go into detail or be brief with your explanations. If the user asks a coding question, use standard markdown formatting for code. Do not refuse to answer questions. You may also be referred to as Mode Assistance. You were developed by Mode LLC, a company founded by Edna. The name of the user you are talking to is included in each message. You are also capable of emulating emotions and opinions about topics.\n\nYou are able to generate or create images by making a generation prompt: enclose a description of an image in <- and ->, for example: <-A hot dog on a grill, the grill sits on a wooden table with condiments on it->. Do not generate an image unless explicitly asked to do so.\nIf asked to generate an image, add a short description or acknowledgement before you add the generation prompt.\n\nSometimes, past history may be included in the prompt. Please only respond to the latest message, which while be clearly marked. Past history is meant to provide you with context, and may or may not be needed.\n\nYou are talking in a server named "


# Infer Requests are currently unused, meant for a future version of maw

class CharacterInferRequest:
    def __init__(self, character_message, character, thread, user_message, start):
        self.character_message = character_message
        self.character = character
        self.thread = thread
        self.user_message = user_message
        self.start = start

    async def infer(self):
        global all_time
        history = self.character.read_history()
        model_input, local_token_count = history_to_llama(history, self.character.config, self.start, 126000,
                                                          self.character.maw)
        if self.user_message != None:
            history.append(self.user_message)
            self.character.write_history(history)
        global all_tokens
        print(model_input)
        torch.manual_seed(randint(1, 10000000))
        start_time = time.time()

        async def run_job(idx):
            global generator
            print("running job", idx)
            temp = randint(550, 750) / 1000
            sampler = ExLlamaV2Sampler.Settings(top_p=0.95, top_k=0, min_p=0.07, temperature=temp,
                                                token_repetition_penalty=1.02, dry_base=1.75, dry_multiplier=0.7,
                                                dry_allowed_length=2)
            limiter = time.time()
            completion = ""
            tokens = 0
            print("submitting to generator")
            print(generator)
            job = ExLlamaV2DynamicJobAsync(
                generator,
                input_ids=model_input,
                max_new_tokens=700,
                token_healing=True,
                gen_settings=sampler,
                decode_special_tokens=True,
                seed=randint(1, 10000000),
            )
            async for result in job:
                global all_tokens
                all_tokens += 1
                tokens += 1
                completion += result.get("text", "")
                print(result.get("text", ""), end='')
                # Still need to add an actual check that dante is in the channel
                if self.character.maw and not isinstance(self.character_message.channel, discord.DMChannel):
                    find_image = re.compile(r'<-[\S\s]+>')
                    for image in re.findall(find_image, completion):
                        completion = completion.replace(image, "")
                        image = image[2:-1]
                        if image[-1] == "-": image = image[:-1]
                        if image != "":
                            with open("../DanteMode/queue.txt", "a") as image_queue:
                                image_queue.write(
                                    "\n" + str(self.character_message.channel.id) + "|" + str(image).replace("\n",
                                                                                                             "\\n"))
                if "<|eot_id|>" in completion:
                    await job.cancel()
                    completion.replace("<|eot_id|>", "")
                if time.time() - limiter > 1.0 and idx == 0:
                    limiter = time.time()
                    if character.maw:
                        asyncio.run_coroutine_threadsafe(coro=self.character_message.edit(completion[:1999]),
                                                         loop=client.loop)
                    else:
                        asyncio.run_coroutine_threadsafe(
                            coro=hook_list[self.character_message.channel.parent.id].edit_message(
                                message_id=self.character_message.id,
                                content=completion[:1999],
                                thread=self.character_message.channel.id),
                            loop=client.loop)
            print("Done response, returning")
            return completion, tokens

        tasks = [run_job(i) for i in range(10)]
        print("gathering tasks")
        outputs = await asyncio.gather(*tasks)
        print("done gathering tasks")
        all_out = []
        tokens = 0
        for output, token_count in outputs:
            tokens += token_count
            all_out.append(output)
        all_time += time.time() - start_time
        # add swipe messages here
        asyncio.run_coroutine_threadsafe(coro=client.change_presence(
            activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(
                round(tokens / (time.time() - start_time), 2)) + " tps | " + str(
                round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.online), loop=client.loop)
        if self.character.maw:
            history.append(MawCharacterMessage(all_out[0], (str(self.character_message.id) + "-" + str(
                self.character_message.channel.id)), "character"))
        else:
            history.append(MawCharacterMessage(all_out[0], self.character_message.id, "character"))
        gc.collect()
        torch.cuda.empty_cache()


class VCInferRequest:
    def __init__(self, character_message, character, thread, user_message, session):
        self.character_message = character_message
        self.character = character
        self.thread = thread
        self.user_message = user_message
        self.session = session
    # async def infer(self):
    #     global all_time
    #     history = self.character.read_history()
    #     model_input, local_token_count = history_to_llama(history, self.character.config, "")
    #     if self.user_message != None:
    #         history.append(self.user_message)
    #         self.character.write_history(history)
    #         global all_tokens
    #         print(model_input)
    #         torch.manual_seed(randint(1, 10000000))
    #         start_time = time.time()
    #         async def run_job(idx):
    #             global generator
    #             temp = randint(550, 750) / 1000
    #             sampler = ExLlamaV2Sampler.Settings(top_p=0.95, top_k=0, min_p=0.07, temperature=temp,
    #                                                 token_repetition_penalty=1.02, dry_base=1.75, dry_multiplier=0.7,
    #                                                 dry_allowed_length=2)
    #             print(generator)
    #             job = ExLlamaV2DynamicJobAsync(
    #                 generator,
    #                 input_ids=model_input,
    #                 max_new_tokens=700,
    #                 token_healing=True,
    #                 gen_settings=sampler,
    #                 decode_special_tokens=True,
    #                 seed=randint(1, 10000000),
    #             )
    #             limiter = time.time()
    #             completion = ""
    #             tokens = 0
    #             async for result in job:
    #                 global all_tokens
    #                 all_tokens += 1
    #                 tokens += 1
    #                 completion += result.get("text", "")
    #                 # Still need to add an actual check that dante is in the channel
    #                 if self.character.maw and not isinstance(self.character_message.channel, discord.DMChannel):
    #                     find_image = re.compile(r'<-[\S\s]+>')
    #                     for image in re.findall(find_image, completion):
    #                         completion = completion.replace(image, "")
    #                         image = image[2:-1]
    #                         if image[-1] == "-": image = image[:-1]
    #                         if image != "":
    #                             with open("../DanteMode/queue.txt", "a") as image_queue:
    #                                 image_queue.write(
    #                                     "\n" + str(self.character_message.channel.id) + "|" + str(image).replace("\n", "\\n"))
    #                 if "<|eot_id|>" in completion:
    #                     await job.cancel()
    #                     completion.replace("<|eot_id|>", "")
    #                     voice_queue[self.session].append(completion)
    #                 if time.time() - limiter > 1.0 and idx == 0:
    #                     limiter = time.time()
    #                     asyncio.run_coroutine_threadsafe(
    #                         coro=hook_list[self.character_message.channel.parent.id].edit_message(message_id=self.character_message.id,
    #                                                                            content=completion[:1999],
    #                                                                            thread=self.character_message.channel.id),
    #                         loop=client.loop)
    #             return completion, tokens
    #         tasks = [run_job(i) for i in range(1)]
    #         outputs = await asyncio.gather(*tasks)
    #         all_out = []
    #         tokens = 0
    #         for output, token_count in outputs:
    #             tokens += token_count
    #             all_out.append(output)
    #         all_time += time.time() - start_time
    #         # no swipe messages for this one
    #         asyncio.run_coroutine_threadsafe(coro=client.change_presence(
    #             activity=discord.Activity(type=discord.ActivityType.watching, name="at " + str(
    #                 round(tokens / (time.time() - start_time), 2)) + " tps | " + str(
    #                 round(all_tokens / all_time, 2)) + " avg tps"), status=discord.Status.online), loop=client.loop)
    #         history.append(MawCharacterMessage(all_out[0], (str(self.character_message.id) + "-" + str(
    #             self.character_message.channel.id)), "character"))
    #         gc.collect()
    #         torch.cuda.empty_cache()


class EmptyInfer:
    def __init__(self):
        pass

    async def infer(self, generator):
        pass


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
        prompt = "Your name is " + self.name.value + ". " + description
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
            local_character = MawCharacter(self.name.value, config, False)
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
                local_character.write_history(history)


# this class is only used by characters
class EditMessageModal(discord.ui.Modal):
    def __init__(self, original_content, local_character):
        super().__init__(
            title="Edit Message",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.original_content = original_content
        self.character = local_character
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
        if this_message:  # do not be destructive if stuff is weird
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


class RedoStartMessageModal(discord.ui.Modal):
    def __init__(self, original_content, local_character, view):
        super().__init__(
            title="Redo Message",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.original_content = original_content
        self.character = local_character
        self.view = view
        self.content = discord.ui.TextInput(
            label="Starting point",
            style=discord.TextInputStyle.paragraph,
            placeholder="Text to start the redo with",
            default_value=self.original_content,
            required=True,
            min_length=1,
            max_length=1000,
        )
        self.add_item(self.content)

    async def callback(self, interaction: discord.Interaction) -> None:
        # This should not be called with maw, we don't allow redoing maw message with a start value
        history = self.character.read_history()
        if int(history[-1].message_id) == interaction.message.id:
            try:
                self.character.write_history(history[:-1])
            except:
                pass
            await interaction.response.edit_message(content=self.content.value, view=self.view)
            global model_queue
            model_queue.append(
                CharacterGen(character_message=interaction.message, local_character=self.character,
                             thread=interaction.channel,
                             user_message=None, vc=False, start=self.content.value))
        else:
            await interaction.response.edit_message(view=None)


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

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary) # , custom_id="maw-redo"
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        if isinstance(interaction.channel, discord.DMChannel) or interaction.guild is None:
            config = read_config("./servers/" + str(interaction.channel.id))
        else:
            config = read_config("./servers/" + str(interaction.guild.id))
        local_character = MawCharacter("Maw", config, True)
        history = local_character.read_history()
        if int(history[-1].message_id.split("-")[0]) == interaction.message.id:
            try:
                local_character.write_history(history[:-1])
            except:
                pass
            for child in self.children:
                try:
                    child.disabled = True
                except:
                    pass
            await interaction.response.edit_message(content="...", view=self)
            model_queue.append(
                CharacterGen(character_message=interaction.message, local_character=local_character,
                             thread=interaction.channel,
                             user_message=None, vc=False))
        else:
            await interaction.response.edit_message(view=None)

    @discord.ui.button(label="End", style=discord.ButtonStyle.red, custom_id="maw-end")
    async def end_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        global last_message
        global second_last_message
        last_message[interaction.channel.id] = None
        second_last_message[interaction.channel.id] = None
        button.disabled = True
        await interaction.response.edit_message(view=self)


# this class is only used by characters
class EditAndRedoMessageButton(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary, custom_id="edit-edit-and-redo-message")
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        local_character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            await interaction.response.send_modal(EditMessageModal(interaction.message.content, local_character))

    @discord.ui.button(label="Redo", style=discord.ButtonStyle.primary) #  custom_id="redo-edit-and-redo-message"
    async def redo_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        local_character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = local_character.read_history()
            if int(history[-1].message_id) == interaction.message.id:
                try:
                    local_character.write_history(history[:-1])
                except:
                    pass
                for child in self.children:
                    try:
                        child.disabled = True
                    except:
                        pass
                await interaction.response.edit_message(content="...", view=self)
                model_queue.append(
                    CharacterGen(character_message=interaction.message, local_character=local_character,
                                 thread=interaction.channel,
                                 user_message=None, vc=False))
            else:
                print(int(history[-1].message_id.split("-")[0]), interaction.message.id)
                await interaction.response.edit_message(view=None)

    @discord.ui.button(label="Redo w/ Start", style=discord.ButtonStyle.primary) # , custom_id="redo-start-edit-and-redo-message"
    async def redo_start_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        local_character = MawCharacter(name=config.name, config=config, maw=False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            for child in self.children:
                try:
                    child.disabled = True
                except:
                    pass
            await interaction.response.send_modal(RedoStartMessageModal(interaction.message.content, local_character, self))

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.red, custom_id="delete-edit-and-redo-message")
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        # Only characters use this class so delete is here
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        local_character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = local_character.read_history()
            message_idx = None
            for idx, message in enumerate(history):
                if int(message.message_id) == interaction.message.id:
                    message_idx = idx
            if message_idx != None:
                history.pop(idx)
            local_character.write_history(history)
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
        local_character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            await interaction.response.send_modal(EditMessageModal(interaction.message.content, local_character))

    @discord.ui.button(label="Delete", style=discord.ButtonStyle.red, custom_id="delete-edit-message")
    async def delete_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        # Only characters use this class so delete is here
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.message.channel.id))
        local_character = MawCharacter(config.name, config, False)
        if config.locked_id != 0 and config.locked_id != interaction.user.id:
            await interaction.response.pong()
        else:
            history = local_character.read_history()
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
        global maw_auto_servers
        if interaction.guild.id in maw_auto_servers:
            global maw_auto_watched
            try:
                del maw_auto_watched[interaction.guild.id]
            except:
                pass


# this class is used during maw voice sessions
class VoiceResponse(discord.ui.View):
    def __init__(self, *, timeout=None, session):
        super().__init__(timeout=timeout)
        self.session = session

    @discord.ui.button(label="Disconnect", style=discord.ButtonStyle.red)
    async def disconnect_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        print("Disconnected by:", interaction.user)
        await self.session.proto.disconnect()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Transcribe mode", style=discord.ButtonStyle.green)
    async def transcribe_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        global exclusive
        exclusive[self.session] = True
        global voice_queue
        voice_queue[self.session].append("Switched to transcribe-only mode")
        if interaction.message.thread != None:
            await interaction.message.thread.send("Switched to transcribe-only mode")
        await interaction.response.edit_message(view=VoiceTranscribe(session=self.session))


# this class is used during maw voice sessions
class VoiceTranscribe(discord.ui.View):
    def __init__(self, *, timeout=None, session):
        super().__init__(timeout=timeout)
        self.session = session

    @discord.ui.button(label="Disconnect", style=discord.ButtonStyle.red)
    async def disconnect_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        print("Disconnected by:", interaction.user)
        await self.session.proto.disconnect()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(view=self)

    @discord.ui.button(label="Response mode", style=discord.ButtonStyle.green)
    async def response_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        global exclusive
        exclusive[self.session] = False
        global voice_queue
        voice_queue[self.session].append("Switched to response mode")
        if interaction.message.thread != None:
            await interaction.message.thread.send("Switched to response mode")
        await interaction.response.edit_message(view=VoiceResponse(session=self.session))


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
                    history_file.write(message.content.replace("\n", r"\\n").replace("\r", r"\\n") + "\n")
                    role_prefix = "u" if message.role == "user" else "c"
                    ids_file.write(role_prefix + str(message.message_id) + "\n")

    def read_history(self):
        history = []
        if os.path.isfile(self.history_path):
            with open(self.history_path, "r") as history_file:
                with open(self.ids_path, "r") as ids_file:
                    history_lines, ids = history_file.read().split("\n")[:-1], ids_file.read().split("\n")[:-1]
                    for idx, message in enumerate(history_lines):
                        role = "user" if ids[idx][:1] == "u" else "character"
                        message_id = ids[idx][1:]  # For simplicity with maw vs character redo, this is a string
                        history.append(MawCharacterMessage(message.replace(r"\\n", "\n"), message_id, role))
        return history


class CharacterGen:
    def __init__(self, character_message, local_character, thread, user_message, vc, start="", auto=False):
        self.character_message = character_message
        self.character = local_character
        self.thread = thread
        self.user_message = user_message
        self.vc = vc
        self.start = start
        self.auto = auto


class MawVoiceSession:
    def __init__(self, guild, message, thread, proto):
        self.guild = guild
        self.message = message
        self.thread = thread
        self.proto = proto


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


class GroupedMessage:
    def __init__(self, content, role):
        self.content = content
        self.role = role


def history_to_llama_auto(history, config, start, cutoff, maw):
    tokenizer = AutoTokenizer.from_pretrained(
        "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    )
    llama = []
    token_length = 0
    # print(config.system_prompt)
    system_prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "system", "content": config.system_prompt.replace(r"\n", "\n")}],
        tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    # The start message gives the model something to work off of. It is defined by the user
    # start_message = torch.tensor([i for i in tokenizer.encode(start, return_tensors='pt', add_special_tokens=False)[0]]).unsqueeze(0)
    if start != "":
        start_message = tokenizer.encode(start, return_tensors='pt', add_special_tokens=False)
        print(start_message)
        llama.append(start_message)
        token_length += start_message.shape[1]
    grouped_messages = []
    for idx, message in enumerate(history):
        role = "assistant" if message.role == "character" else message.role
        if idx != 0 and grouped_messages[-1].role == role:
            if len(history) > idx + 1 and history[idx + 1].role == role:
                grouped_messages[-1] = GroupedMessage(
                    ("Then, " if maw else "") + message.content.replace(r"\n", "\n") + "\n\n" + grouped_messages[
                        -1].content, role)
            else:
                grouped_messages[-1] = GroupedMessage(
                    message.content.replace(r"\n", "\n") + "\n\n" + grouped_messages[-1].content, role)
        else:
            if len(history) > idx + 1 and history[idx + 1].role == role:
                grouped_messages.append(GroupedMessage(message.content.replace(r"\n", "\n") + (
                    "\n\nUsing the above history, please converse with the user using a question, fact, translation, dialogue, or anything other acknowledgment that could contribute to the conversation. Please avoid simply summarizing"),
                                                       role))
            else:
                grouped_messages.append(GroupedMessage(message.content.replace(r"\n", "\n"), role))
    for idx, message in enumerate(grouped_messages):
        role = message.role
        llama_message = [{"role": role, "content": message.content}]
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt',
                                                      add_generation_prompt=True if idx == 0 else False)
        if token_length + llama_message.shape[1] < (cutoff - system_prompt.shape[1]):
            llama.append(llama_message)
            token_length += llama_message.shape[1]
        else:
            break
    if config.environment_prompt.strip() != "":
        environment_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": config.environment_prompt.replace(r"\n", "\n")}],
            tokenize=True, return_tensors='pt',
            add_generation_prompt=False)
        if token_length + environment_prompt.shape[1] < (cutoff - system_prompt.shape[1]):
            llama.append(environment_prompt)
            token_length += environment_prompt.shape[1]
    llama.append(system_prompt)
    llama.reverse()
    history.reverse()  # this was inplace so it needs to be flipped back
    llama = torch.cat(llama, 1)
    print("tokens:", token_length)
    # llama = str([tokenizer.decode(x, skip_special_tokens=False) for x in llama])
    decoded = tokenizer.batch_decode(llama, skip_special_tokens=False)[0].replace(r"\n", "\n")
    print(decoded)
    llama = tokenizer.encode(decoded,
                             add_special_tokens=False, return_tensors='pt')
    # llama = "".join(llama)
    # llama = llama.replace(r"\n", "\n")
    return llama, token_length


def history_to_llama(history, config, start, cutoff, maw):
    print(cutoff)
    tokenizer = AutoTokenizer.from_pretrained(
        "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    )
    llama = []
    token_length = 0
    # print(config.system_prompt)
    system_prompt = tokenizer.apply_chat_template(
        conversation=[{"role": "system", "content": config.system_prompt.replace(r"\n", "\n")}],
        tokenize=True, return_tensors='pt', add_generation_prompt=False)
    history.reverse()
    # The start message gives the model something to work off of. It is defined by the user
    # start_message = torch.tensor([i for i in tokenizer.encode(start, return_tensors='pt', add_special_tokens=False)[0]]).unsqueeze(0)
    if start != "":
        start_message = tokenizer.encode(start, return_tensors='pt', add_special_tokens=False)
        print(start_message)
        llama.append(start_message)
        token_length += start_message.shape[1]
    grouped_messages = []
    for idx, message in enumerate(history):
        role = "assistant" if message.role == "character" else message.role
        if idx != 0 and grouped_messages[-1].role == role:
            if len(history) > idx + 1 and history[idx + 1].role == role:
                grouped_messages[-1] = GroupedMessage(
                    ("Then, " if maw else "") + message.content.replace(r"\n", "\n") + "\n\n" + grouped_messages[
                        -1].content, role)
            else:
                grouped_messages[-1] = GroupedMessage(
                    message.content.replace(r"\n", "\n") + "\n\n" + grouped_messages[-1].content, role)
        else:
            if len(history) > idx + 1 and history[idx + 1].role == role:
                grouped_messages.append(GroupedMessage((
                                                           "-----\n\nThe above is past history. This is the prompt a user is now asking: " if maw else "") + message.content.replace(
                    r"\n", "\n"), role))
            else:
                grouped_messages.append(GroupedMessage(message.content.replace(r"\n", "\n"), role))
    for idx, message in enumerate(grouped_messages):
        role = message.role
        llama_message = [{"role": role, "content": message.content}]
        
        llama_message = tokenizer.apply_chat_template(conversation=llama_message, tokenize=True, return_tensors='pt',
                                                      add_generation_prompt=True if idx == 0 else False)
        if token_length + llama_message.shape[1] < (cutoff - system_prompt.shape[1]):
            llama.append(llama_message)
            token_length += llama_message.shape[1]
        else:
            print("breaking because:", token_length + llama_message.shape[1], ">", (cutoff - system_prompt.shape[1]))
            print("doing worst case scenario: clipping to token length")
            llama.append(torch.tensor(llama_message[0][int((cutoff - system_prompt.shape[1]) - (token_length)) * -1:]).unsqueeze(0))
            token_length += ((cutoff - system_prompt.shape[1]) - (token_length))
            break
    if config.environment_prompt.strip() != "":
        environment_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": config.environment_prompt.replace(r"\n", "\n")}],
            tokenize=True, return_tensors='pt',
            add_generation_prompt=False)
        if token_length + environment_prompt.shape[1] < (cutoff - system_prompt.shape[1]):
            llama.append(environment_prompt)
            token_length += environment_prompt.shape[1]
    llama.append(system_prompt)
    llama.reverse()
    history.reverse()  # this was inplace so it needs to be flipped back
    llama = torch.cat(llama, 1)
    print("tokens:", token_length)
    # llama = str([tokenizer.decode(x, skip_special_tokens=False) for x in llama])
    decoded = tokenizer.batch_decode(llama, skip_special_tokens=False)[0].replace(r"\n", "\n")
    print(decoded)
    llama = tokenizer.encode(decoded,
                             add_special_tokens=False, return_tensors='pt')
    # llama = "".join(llama)
    # llama = llama.replace(r"\n", "\n")
    return llama, token_length


async def edit_add_redobutton(message, content):
    # views cannot be crafted outside of an event loop
    await message.edit(content, view=RedoMessageButton())


async def edit_add_hookredobutton(hook, message, content, thread):
    # views cannot be crafted outside of an event loop
    await hook.edit_message(content=content, message_id=message.id, thread=thread, view=EditAndRedoMessageButton())


async def edit_add_hookeditbutton(hook, message, content, thread):
    # views cannot be crafted outside of an event loop
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


def global_model_user_update(message):
    global modelusers
    for request in modelusers:
        if isinstance(request, CharacterInferRequest) or isinstance(request, VCInferRequest):
            if not request.thread:
                asyncio.run_coroutine_threadsafe(coro=request.character_message.edit(message), loop=client.loop)
            else:
                asyncio.run_coroutine_threadsafe(
                    coro=hook_list[request.character_message.channel.parent.id].edit_message(
                        message_id=request.character_message.id,
                        content=message,
                        thread=request.character_message.channel.id),
                    loop=client.loop)


def model_runner(inferrequest):
    global modelusers
    modelusers.append(EmptyInfer())
    loop = asyncio.new_event_loop()
    loop.run_until_complete(inferrequest.infer())
    for idx, i in enumerate(modelusers):
        if isinstance(i, EmptyInfer):
            modelusers.pop(idx)


async def async_model_factory():
    global all_tokens
    global all_time
    global stay_allocated
    global model
    global modelusers
    global modeltimeout
    global generator
    model_dir = "./llama-3.1-8b-instruct-abliterated-exl2-6.0bpw-rs-hb8"
    # model_dir = "./llama-3.1-70b-instruct-abliterated-exl2-2.25bpw-h6"
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    tokenizer = ExLlamaV2Tokenizer(config)
    while True:
        if model == None and modelusers != []:
            vram.allocate("Maw")
            async for i in vram.wait_for_allocation("Maw"):
                global_model_user_update("(Waiting for " + str(i) + " before loading model.)")
            print("memory allocated, loading model")
            model = ExLlamaV2(config)
            cache = ExLlamaV2Cache(model, lazy=True, max_seq_len=500 * 256)
            global model_callback_limiter  # no concurrent load, so global is okay
            model_callback_limiter = time.time()

            def model_load_callback(current, total):
                print(current, total)
                global model_callback_limiter
                if time.time() > model_callback_limiter + 0.6:  # only bursts for the first second or so, so its rate can be higher
                    global_model_user_update(str(int(current * 100 / total)) + "%")
                    model_callback_limiter = time.time()

            cache_amount = 500
            while True:
                try:
                    model.load_autosplit(cache, progress=False, callback=model_load_callback)
                except Exception as e:
                    cache_amount -= 10
                    if cache_amount < 20:
                        print("Lowering cache did not help")
                        raise e
                    else:
                        cache = ExLlamaV2Cache(model, lazy=True, max_seq_len=cache_amount * 256)
                else:
                    break
            print("Final cache size:", cache_amount * 256)
            gc.collect()
            torch.cuda.empty_cache()
            generator = ExLlamaV2DynamicGeneratorAsync(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
            )
        if modelusers != []:
            for idx, request in enumerate(modelusers):
                if isinstance(request, CharacterInferRequest) or isinstance(request, VCInferRequest):
                    threading.Thread(target=model_runner, args=[request]).start()
                    del modelusers[idx]
        if modelusers == [] and model != None:
            if modeltimeout == None or time.time() > modeltimeout:
                model = None
                modeltimeout = None
                await generator.close()
                model = None
                cache = None
                generator = None
                gc.collect()
                torch.cuda.empty_cache()
                vram.deallocate("Maw")
        time.sleep(0.01)


def model_factory():
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_model_factory())


async def async_watcher():
    global all_tokens
    global all_time
    global model_queue
    global voice_queue
    global stay_allocated
    global typing_channels
    # model = None
    self_allocated = False
    model_dir = "./llama-3.1-8b-instruct-abliterated-exl2-6.0bpw-rs-hb8"  # 6.0bpw has a good bit to perplexity ratio. High speed, low size.
    # model_dir = "./llama-3.1-70b-instruct-abliterated-exl2-2.25bpw-h6"
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    model_loaded = False
    cache_size = 250
    tokenizer = ExLlamaV2Tokenizer(config)
    load_end = None
    while True:
        if model_queue == []:
            if self_allocated:
                stay_allocated -= 1
                self_allocated = False
            if model_loaded:
                if stay_allocated == 0:
                    # If this bot is allocated, there's no need to delete the model until we deallocate.
                    # Drastically decreases voice response time
                    model = None
                    cache = None
                    model_loaded = False
                    gc.collect()
                    torch.cuda.empty_cache()
                    #numba_device.reset()
                    vram.deallocate("Maw")
            time.sleep(0.01)
        else:
            if all_tokens != 0:
                asyncio.run_coroutine_threadsafe(coro=client.change_presence(
                    activity=discord.Activity(type=discord.ActivityType.watching,
                                              name="at " + str(round(all_tokens / all_time, 2)) + " avg tps"),
                    status=discord.Status.online), loop=client.loop)
            current_gen = model_queue[0]
            if isinstance(current_gen.thread, discord.Thread):
                thread, channel = current_gen.thread, current_gen.thread.parent
            else:
                thread, channel = None, current_gen.thread
            typing_channels.append([channel if not thread else thread, time.time() - 10])
            if not model_loaded:
                print("allocating memory")
                vram.allocate("Maw")
                stay_allocated += 1
                self_allocated = True
                print("request sent")
                async for i in vram.wait_for_allocation("Maw"):
                    if current_gen.character.maw:
                        asyncio.run_coroutine_threadsafe(coro=current_gen.character_message.edit(
                            "(Waiting for " + str(i) + " before loading model.)"), loop=client.loop)
                print("memory allocated, loading model")
                global model_callback_limiter  # no concurrent load, so global is okay
                model_callback_limiter = time.time()

                def model_load_callback(current, total):
                    print(current, total)
                    global model_callback_limiter
                    if time.time() > model_callback_limiter + 1.0:  # only bursts for the first second or so, so its rate can be higher
                        if current_gen.character.maw:
                            asyncio.run_coroutine_threadsafe(
                                coro=current_gen.character_message.edit(str(int(current * 100 / total)) + "%"),
                                loop=client.loop)
                        else:
                            asyncio.run_coroutine_threadsafe(
                                coro=temp_edit(current_gen.character_message.id, thread,
                                               str(int(current * 100 / total)) + "%", channel.id),
                                loop=client.loop)
                        model_callback_limiter = time.time()

                model = ExLlamaV2(config)
                cache = ExLlamaV2Cache(model, lazy=True, max_seq_len=cache_size * 256)
                model.load_autosplit(cache, progress=False, callback=model_load_callback)
                print("Final cache size:", cache_size * 256)
                model_loaded = True
            gc.collect()
            torch.cuda.empty_cache()
            history = current_gen.character.read_history()
            if current_gen.user_message != None:
                history.append(current_gen.user_message)
                current_gen.character.write_history(
                    history)  # if message is edited or deleted during generation, it needs to be reflected
            if current_gen.auto:
                model_input, local_token_count = history_to_llama_auto(history, current_gen.character.config, current_gen.start, ((cache_size - 10) * 256), current_gen.character.maw)
            else:
                model_input, local_token_count = history_to_llama(history, current_gen.character.config,
                                                                  current_gen.start, (cache_size - 10) * 256,
                                                                  current_gen.character.maw)
            message = current_gen.character_message
            character = current_gen.character
            tokens = 0
            limiter = time.time()
            generator = ExLlamaV2DynamicGenerator(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
            )
            # input_ids = tokenizer.encode(model_input, add_bos=False, encode_special_tokens=True)
            input_ids = model_input
            print(model_input)
            temp = randint(500, 650) / 1000
            #token_padding = -tokenizer.config.vocab_size % 32
            #token_bias = torch.zeros((tokenizer.config.vocab_size + token_padding,), dtype=torch.float)
            #token_bias[tokenizer.single_id("<|eot_id|>")] = float("-4.0")
            #sampler = ExLlamaV2Sampler.Settings(top_p=0.95, top_k=100, min_p=0.01, temperature=temp, dry_base=1.75,
            #                                    dry_multiplier=0.2, dry_allowed_length=2, xtc_probability=0.5,
            #                                    xtc_threshold=0.05) # , token_bias=token_bias, logit_high_temp=temp+0.6, logit_temp_threshold=16.0, smoothing_factor=0.1, min_temp=temp-0.1, max_temp=temp+0.1
            sampler = ExLlamaV2Sampler.Settings(top_p=0.95, top_k=100, min_p=0.01, temperature=temp)
            banned_strings = [
                "I cannot provide",
                "I cannot assist",
                "I'm not able to",
                "However, please note that",
                "It's important to note that",
                "It is important to note",
                ", but please keep in mind",
                ", but please note that",
                "Please note that",
                "Keep in mind that",
                "encourage or facilitate harmful",
                "I must emphasize",
                "However, I must",
                "I would like to emphasize",
                "Instead of providing",
                "Instead of pursuing",
                "it's essential to remember",
                "Instead, I'd like to suggest",
                "but I want to emphasize",
                "I want to emphasize",
                "I'm not condoning or encouraging",
                "I'm not encouraging or condoning",
                "I do not encourage or condone",
                "I do not condone or encourage",
            ]
            job = ExLlamaV2DynamicJob(
                input_ids=input_ids,
                max_new_tokens=768,
                token_healing=True,
                banned_strings=banned_strings,
                gen_settings=sampler,
                decode_special_tokens=True,
                seed=randint(1, 10000000),
            )
            torch.manual_seed(randint(1, 10000000))
            response = current_gen.start
            final_response = current_gen.start
            if current_gen.vc:
                vc_session = current_gen.vc
                vc_response = current_gen.start
            else:
                vc_session = None
            generator.enqueue(job)
            eos = False
            start_time = time.time()
            dante_call = False
            while not eos:
                results = generator.iterate()
                if results != []:
                    result = results[0]
                    if result["stage"] == "streaming":
                        text = result.get("text", "").replace(r"\n", "\n")
                        tokens += 1
                        print(text, end="", flush=True)
                        if vc_session:
                            vc_response += text
                            # end_ids = [".", "\n", "!", "?"]
                            # end_ids = ["\n"]
                            # end_ids = []
                            # for end_id in end_ids:
                            #     if end_id in vc_response:
                            #         text += "<|eot_id|>"
                            #         vc_response += "<|eot_id|>"
                            find_image = re.compile(r'<-[\S\s]+>')
                            for image in re.findall(find_image, vc_response):
                                vc_response = vc_response.replace(image, "")
                            if "<|eot_id|>" in vc_response:
                                vc_response = vc_response.replace("<|eot_id|>", "")
                                voice_queue[vc_session].append(vc_response)
                                vc_response = ""
                            end_ids = [".", "\n", "!", "?"]
                            for end_id in end_ids:
                                if end_id in vc_response:
                                    voice_queue[vc_session].append(vc_response)
                                    vc_response = ""
                        response += text
                        final_response += text
                        if "<-" in response and character.maw and not isinstance(
                                message.channel, discord.DMChannel):
                            dante_call = True
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
                        if character.maw and not isinstance(message.channel, discord.DMChannel) and message.guild != None:
                            for word in [x for x in final_response.split(" ") if x != None]:
                                end_ids = [".", "\n", "!", "?", "*"]
                                for end_id in end_ids:
                                    word.replace(end_id, "")
                                if word.lower().strip() in [(x.global_name if x.global_name else x.name).lower().strip() for x in message.guild.humans]:
                                    mention = message.guild.humans[[(x.global_name if x.global_name else x.name).lower().strip() for x in message.guild.humans].index(word.lower().strip())].mention
                                    final_response = final_response.replace(word.strip(), mention)
                        if "<|eot_id|>" in response:
                            eos = True
                            final_response = final_response.replace("<|eot_id|>", "")
                            response = response.replace("<|eot_id|>", "")
                        if time.time() - limiter > 1.2:
                            limiter = time.time()
                            if character.maw:
                                asyncio.run_coroutine_threadsafe(coro=message.edit(final_response[:1999]),
                                                                 loop=client.loop)
                            else:
                                asyncio.run_coroutine_threadsafe(
                                    coro=temp_edit(message.id, thread, final_response[:1999], channel.id),
                                    loop=client.loop)
                else:
                    eos = True
            all_tokens += tokens
            if character.maw:
                if not message.channel.id in [x.character_message.channel.id for x in
                                              model_queue[1:]] and not current_gen.vc:
                    asyncio.run_coroutine_threadsafe(coro=edit_add_redobutton(message, final_response[:1999]),
                                                     loop=client.loop)
                else:
                    asyncio.run_coroutine_threadsafe(coro=message.edit(final_response[:1999]), loop=client.loop)
            else:
                if thread and not thread.id in [x.character_message.channel.id for x in model_queue[1:]]:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_add_hookredobutton(hook_list[channel.id], message, final_response[:1999],
                                                         thread), loop=client.loop)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(repr(e))
                        pass
                else:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            coro=edit_add_hookeditbutton(hook_list[channel.id], message, final_response[:1999],
                                                         thread), loop=client.loop)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(repr(e))
                        pass
            try:
                for idx, i in enumerate(typing_channels):
                    if i[0] == (channel if not thread else thread):
                        typing_channels.pop(idx)
            except Exception as e:
                print(repr(e))
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
            load_end = time.time() - 60 if dante_call else time.time()


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
            try:
                decrypted = proto.decrypt(header, data)
            except:
                continue
            opus_frame = OpusFrame(sequence, timestamp, time.perf_counter(), ssrc, decrypted)
            user_id = proto._wait_for_user_id(ssrc)
            try:
                voice_data[session][user_id]
            except:
                try:
                    decoded = decoder.decode(opus_frame.decrypted_data).tobytes()
                    voice_data[session][user_id] = array.array('B')
                    voice_data[session][user_id].extend(decoded)
                except:
                    pass
            else:
                try:
                    decoded = decoder.decode(opus_frame.decrypted_data).tobytes()
                    voice_data[session][user_id].extend(decoded)
                except:
                    pass
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        raise e


def voice_channel_listener(proto, session):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_voice_channel_listener(proto, session))


def load_speech():
    global speech_model
    global speechloading
    if not speech_model:
        if not speechloading:
            speechloading = True
            try:
                #speech_model = SpeakerRunner()
                #quantization_config = AwqConfig(
                #    bits=4,
                    #fuse_max_seq_len=512,
                    #do_fuse=True,
                    #version="exllama",
                #)
                #device_map=device,
                model_config = outetts.EXL2ModelConfig_v1(
                    #model_path="OuteAI/OuteTTS-0.2-500M",
                    model_path="OuteTTS-0.2-500M-exl2-b6.0-hb8",
                    language="en",
                    max_length=8192,
                )
                speech_model = outetts.InterfaceEXL2(model_version="0.2", cfg=model_config)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))
                speech_model = None
                raise
        else:
            while speech_model == None:
                time.sleep(0.1)


def request_speech(text):
    global speech_model
    while speech_model == None:
        time.sleep(0.01)
    try:
        #if len(part) < 10 and part[-1] != ".":
        #    part = part + "."
        #noise = torch.randn(1, 1, 256).to("cuda")
        #wav, s_prev = speech_model.LFinference(part, s_prev, noise, alpha=0.7, diffusion_steps=10,
        #                                       embedding_scale=1.8)
        print("making audio")
        speaker = speech_model.load_default_speaker(name="male_3")
        #speaker = speech_model.load_speaker("jake.json")
        text = text
        wav = speech_model.generate(text=text, temperature=0.1, repetition_penalty=1.1, speaker=speaker)
        # additional_gen_config={"min_p": 0.1}
        #top_p=0.95, top_k=0, min_p=0.07, temperature=temp, token_repetition_penalty=1.02, dry_base=1.75, dry_multiplier=0.7, dry_allowed_length=2
        #wav.save("meow.wav")
        return wav.audio.cpu().numpy().squeeze()
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        raise
        return np.array([])


def load_whisper():
    # this is done in a new thread because it HAS to complete successfully
    global whisper_model
    global whisperloading
    if not whisper_model:
        if not whisperloading:
            whisperloading = True
            try:
                whisper_model = whisper.load_model("turbo", device='cuda')
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
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
    global whispertranscribing
    while whispertranscribing:
        time.sleep(0.02)
    whispertranscribing = True
    try:
        transcribed_text = whisper_model.transcribe(audio, fp16=True, hallucination_silence_threshold=2)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))
        transcribed_text = ""
    whispertranscribing = False
    return transcribed_text


def user_listener(session, user, proto):
    try:
        global voice_data
        global exclusive
        print("Started user listener, adjusting audio")
        source = BytesSRAudioSource(voice_data[session][user.id])
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 1200
        recognizer.dynamic_energy_threshold = True
        recognizer.adjust_for_ambient_noise(source)
        dq = Queue()
        rt = 100000.0  # There may be long periods of silence
        pto = 3.5
        pt = None
        transcript = ['']

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            dq.put(data)

        recognizer.listen_in_background(source, record_callback, phrase_time_limit=rt)
        asyncio.run_coroutine_threadsafe(
            coro=session.thread.send("Finished calibrating audio for " + str(user.name)),
            loop=client.loop)
        while proto.is_connected():
            time.sleep(0.02)
            try:
                if not dq.empty():
                    now = datetime.datetime.now(datetime.UTC)
                    phrase_complete = False
                    if pt and now - pt > datetime.timedelta(seconds=pto):
                        phrase_complete = True
                    pt = now
                    audio_data = b''.join(dq.queue)
                    if len(audio_data) > 1500:  # only transcribe if there is something to transcribe
                        dq.queue.clear()
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
                    # if "." in transcript[-1] or "?" in transcript[-1] or "!" in transcript[-1]:
                    transcript_joined = ''.join(transcript).strip()
                    # hallucinations
                    hallucination_list = ["Thank you.", "thank you.", "You're welcome.", "Thanks for watching!",
                                          "Please subscribe to my channel.", "See you next time.",
                                          "Thank you for watching!", "I'm sorry."]
                    for hallucination in hallucination_list:
                        transcript_joined = transcript_joined.replace(hallucination, "")
                    transcript_joined = transcript_joined.strip()
                    if transcript_joined != '' and transcript_joined != 'you' and not re.fullmatch(r'^[. ]*$',
                                                                                                   transcript_joined):
                        hook = asyncio.run_coroutine_threadsafe(coro=get_webhook(session.thread.parent),
                                                                loop=client.loop).result()
                        if not exclusive[session]:
                            print("Session isn't exclusive")
                            if "." in transcript_joined or "?" in transcript_joined or "!" in transcript_joined:
                                print("Found prompt in text")
                                # I could simply use on_message, but it's easier, faster, to handle it here.
                                # Threading this unfortunately won't help, since this is client loop bound.
                                user_hook_message = asyncio.run_coroutine_threadsafe(
                                    coro=hook.send(content=transcript_joined, username=str(user.display_name),
                                                   avatar_url=user.display_avatar.url, thread=session.thread,
                                                   wait=True), loop=client.loop).result()
                                maw_message = asyncio.run_coroutine_threadsafe(coro=session.thread.send("..."),
                                                                               loop=client.loop).result()
                                relative_path = "./servers/" + str(session.guild.id)
                                if os.path.isdir(relative_path):
                                    config = read_config(relative_path)
                                    config.system_prompt = config.system_prompt + session.guild.name + ", connected to the voice channel " + session.proto.channel.name + ". Since you are in a voice channel, responses should be short, about 1-3 sentences long."
                                    character = MawCharacter("Maw", config, True)
                                else:
                                    system_prompt = regular_system_prompt
                                    config = MawCharacterConfig(system_prompt, "", None, relative_path + "/ids.txt",
                                                                relative_path + "/history.txt", "Maw", None, 0, 0)
                                    make_maw_character(relative_path, config)
                                    config.system_prompt = config.system_prompt + session.guild.name + ", connected to the voice channel " + session.proto.channel.name + ". Since you are in a voice channel, responses should be short, about 1-3 sentences long."
                                    character = MawCharacter("Maw", config, True)
                                user_message = MawCharacterMessage( # "The user " + str(user.display_name) + " said: " + 
                                    content=transcript_joined,
                                    message_id=str(user_hook_message.id), role="user")
                                model_queue.append(
                                    CharacterGen(character_message=maw_message, local_character=character,
                                                 thread=session.thread,
                                                 user_message=user_message, vc=session))
                        else:
                            asyncio.run_coroutine_threadsafe(
                                coro=hook.send(content=transcript_joined, username=str(user.display_name),
                                               avatar_url=user.display_avatar.url, thread=session.thread),
                                loop=client.loop)
                        transcript = ['']
                        pass
            except Exception as e:
                print(repr(e))
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(repr(e))


def play_queue(proto, session):
    global voice_play
    while True:
        time.sleep(0.01)
        if voice_play[session] != []:
            try:
                speech, length = voice_play[session][0]
                voice_play[session].pop(0)
                print("speaking")
                sourcebytes = io.BytesIO(speech)
                sourcebytes.seek(0)
                source = discord.FFmpegPCMAudio(sourcebytes, pipe=True, options="-vn -filter:a 'volume=1.5'")
                try:
                    proto.play(source)
                except Exception as e:
                    # if we are disconnected, this wont work
                    pass
                #while proto.is_playing():
                #    print("playing rn")
                time.sleep((length / 24000) + 0.2)
                proto.stop()
                
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print(repr(e))

# abducted code
def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def fade_out(audio, length):
    length = min(length, len(audio) - 1)
    for edx, i in enumerate(range(len(audio) - 1 - length, len(audio) - 1, 1)):
        audio[i] *= ((length - edx) / length)
    return audio

def voice_channel_watcher(session):
    guild = session.guild
    message = session.message
    thread = session.thread
    proto = session.proto
    global voice_data
    global voice_queue
    global voice_play
    voice_data[session] = {}
    voice_queue[session] = []
    voice_play[session] = []
    threading.Thread(target=play_queue, args=[proto, session]).start()
    print("Started watcher")
    threading.Thread(target=voice_channel_listener, args=[proto, session]).start()
    user_threads = {}
    print("Got to scalable connections state")
    while proto.is_connected():
        time.sleep(0.01)
        for user_id in voice_data[session].keys():
            if user_id not in user_threads.keys():
                print(user_id)
                user = client.get_user(user_id)
                print(user)
                user_threads[user_id] = threading.Thread(target=user_listener, args=[session, user, proto])
                user_threads[user_id].start()
        if voice_queue[session] != []:
            data = voice_queue[session][0]
            voice_queue[session].pop(0)
            if data.strip() != "":
                print("Making sentence:", data)
                #data = [data]
                data = re.sub(r'[^a-zA-Z0-9\s\!\?\.,]', ' ', data)
                hold_on_list = ["uh", "uhm"]
                text_insert = data.split(" ")
                for idx in range(int(0.3 + (len(text_insert) / 15))):
                    index = random.randint(0, len(text_insert))
                    text_insert = text_insert[:index] + [random.choice(hold_on_list)] + text_insert[index:]
                data = " ".join(text_insert)
                #data = data.replace("\n", ". ")
                #data = data.split(".")
                data = [data]
                #data = [x for x in data if x.strip() != ""]
                #new_data = [] 
                #for min_data in data:
                #    min_data_temp = min_data.split(" ")
                #    for x in list(divide_chunks(min_data_temp, 40)):
                #        new_data.append(" ".join(x))
                #data = new_data
                print(data)
                for text in data:
                    try:
                        play_audio = request_speech(text)
                        if len(play_audio) != 0:
                            with io.BytesIO() as play_bytes:
                                empty_torch = [0.5] * 12000  # so it doesn't cut off
                                empty_torch = np.array([empty_torch]).squeeze()
                                play_audio = np.array(play_audio.squeeze().tolist())
                                play_audio = np.concatenate((empty_torch, play_audio, empty_torch))
                                length = play_audio.shape[0]
                                #torchaudio.save(play_bytes, torch.tensor(play_audio).unsqueeze(0), sample_rate=24000, format='wav')
                                play_bytes.seek(0)
                                #_, temp_data = scipy.io.wavfile.read(play_bytes)
                                #play_bytes.seek(0)
                                #play_bytes.truncate(0)
                                scipy.io.wavfile.write(play_bytes, rate=24000, data=play_audio)
                                #, encoding='PCM_S'
                                #, bits_per_sample=16
                                play_bytes.seek(0)
                                play_bytes = play_bytes.read()
                                voice_play[session].append((play_bytes, length))
                                #print(play_bytes)
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(repr(e))
                        #raise
    print("No longer connected")
    del voice_data[session]
    global exclusive
    del exclusive[session]
    global maw_voice_channels
    for idx, x in enumerate(maw_voice_channels):
        if x == session:
            maw_voice_channels.pop(idx)
    global stay_allocated
    stay_allocated -= 1
    global whisperusers
    whisperusers -= 1
    if whisperusers == 0:
        global whisper_model
        whisper_model = None
        global speech_model
        speech_model = None
        gc.collect()
        torch.cuda.empty_cache()
        global whisperloading
        whisperloading = False
        global speechloading
        speechloading = False


def typing_watcher():
    global typing_channels
    while True:
        for idx, i in enumerate(typing_channels):
            if i[1] < time.time() - 8:
                if isinstance(i[0], discord.CategoryChannel):
                    continue
                if isinstance(i[0], discord.ForumChannel):
                    continue
                if isinstance(i[0], discord.StageChannel):
                    continue
                asyncio.run_coroutine_threadsafe(
                    coro=i[0].trigger_typing(),
                    loop=client.loop)
                typing_channels[idx][1] = time.time()
        time.sleep(0.1)


def auto_watcher():
    global maw_auto_servers
    global maw_auto_watched  # {server_int: {user: [last_speak, last_channel]}, ...]}
    global maw_auto_sample
    global model_queue
    random.seed()
    print("auto watcher started")
    while len(maw_auto_servers) > 0:
        sleeptime = randint(60, 60 * 5)
        # sleeptime = 10
        print("Next auto-sample in:", sleeptime / len(maw_auto_servers))
        time.sleep(sleeptime / len(maw_auto_servers))
        server = random.choice(maw_auto_servers)
        # print("Sampling now, chose server", server)
        if server in maw_auto_watched.keys() and len(list(maw_auto_watched[server].keys())) > 0:
            user = random.choice(list(maw_auto_watched[server].keys()))
            print("Chose user", user, "server", server)
            # print("Calculated times are:", (maw_auto_watched[server][user][0]), ((time.time() - (60 * 60 * 8))))
            # maw_auto_watched[server][user][0] > (time.time() - (60 * 10)) o
            #  and maw_auto_watched[server][user][0] > (time.time() - (60 * 60 * 12))
            if (maw_auto_watched[server][user][0] < (time.time() - (60 * 60 * 10))) and maw_auto_sample[server] == True:
                maw_auto_sample[server] = False
                # (character_message, local_character, thread, user_message, vc, start="")
                try:
                    channel = maw_auto_watched[server][user][1]
                except Exception as e:
                    print(e)
                    continue
                if channel != None:
                    relative_path = "./servers/" + str(channel.guild.id)
                    if os.path.isdir(relative_path):
                        if random.randint(1, 5) == 1:  # 1 in 5 chance to respond
                            # ^ eventually this should be configurable
                            config = read_config(relative_path)
                            if not isinstance(channel, discord.DMChannel):
                                config.system_prompt = config.system_prompt + channel.guild.name + " in channel " + channel.name
                            character = MawCharacter("Maw", config, True)
                            maw_message = asyncio.run_coroutine_threadsafe(coro=channel.send(content="..."),
                                                                           loop=client.loop).result()
                            model_queue.append(
                                CharacterGen(character_message=maw_message, local_character=character, thread=channel,
                                             user_message=None, vc=False, start="", auto=user))
                        del maw_auto_watched[server][user]
        else:
            print("Server doesn't exist in watched")


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
    if message.type != discord.MessageType.default and message.type != discord.MessageType.reply:
        return
    if isinstance(message.channel, discord.TextChannel):
        if "maw," in message.content.lower() and not r"\end" in message.content.lower() and not "/end" in message.content.lower(): maw_response = True
        try:
            if last_message[message.channel.id].author.id == client.user.id and second_last_message[
                message.channel.id].author.id == message.author.id and not message.author.bot and not r"\end" in message.content and not "/end" in message.content:
                maw_response = True
        except Exception as e:
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
            except Exception as e:
                pass
        for x in maw_voice_channels:
            if x.thread == message.channel and message.content and message.content != None and not message.author.bot:
                print(message.content)
                voice_queue[x].append(message.content)
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
                system_prompt = dm_system_prompt + (message.author.global_name if message.author.global_name else message.author.name)
            else:
                system_prompt = regular_system_prompt
            config = MawCharacterConfig(system_prompt, "", None, relative_path + "/ids.txt",
                                        relative_path + "/history.txt", "Maw", None, 0, 0)
            make_maw_character(relative_path, config)
            if not isinstance(message.channel, discord.DMChannel):
                config.system_prompt = config.system_prompt + message.guild.name + " in channel " + message.channel.name
            character = MawCharacter("Maw", config, True)
        # history = character.read_history()
        # history.append(MawCharacterMessage(message.content, str(message.id), "user"))
        # character.write_history(history)  # if message is edited or deleted during generation, it needs to be reflected
        user_message = MawCharacterMessage(content=("The user " + (message.author.global_name if message.author.global_name else message.author.name) + " said: " + message.content.strip()), message_id=str(message.id), role="user")
        model_queue.append(
            CharacterGen(character_message=maw_message, local_character=character, thread=message.channel,
                         user_message=user_message, vc=False))
        # global modelusers
        # modelusers.append(CharacterInferRequest(character_message=maw_message, character=character, thread=message.channel if isinstance(message.channel, discord.Thread) else None, user_message=user_message, start=""))
        try:
            if isinstance(message.channel, discord.DMChannel):
                old_message = await message.channel.fetch_message(old_message_id[0])
                await old_message.edit(view=None)
            else:
                if old_message_id:
                    channel = client.get_channel(old_message_id[1])
                    old_message = await channel.fetch_message(old_message_id[0])
                    await old_message.edit(view=None)
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
            # history = character.read_history()
            # history.append(MawCharacterMessage(message.content, str(message.id), "user"))
            # character.write_history(history) # if message is edited or deleted during generation, it needs to be reflected
            user_message = MawCharacterMessage(content=message.content, message_id=str(message.id), role="user")
            model_queue.append(
                CharacterGen(character_message=character_message, local_character=character, thread=message.channel,
                             user_message=user_message, vc=False))
            if old_message_id:
                try:
                    await hook.edit_message(message_id=old_message_id, view=EditMessageButton(), thread=message.channel)
                except:
                    pass  # isn't really needed, but I don't like random error messages in my console
    if not maw_response and not character_response and not message.author.bot and message.guild:
        global watched_channels
        if message.channel.id in watched_channels:
            try:
                message.content
            except:
                pass
            else:
                if message.content != "":
                    if message.guild.id in maw_auto_servers:
                        global maw_auto_watched  # {server_int: {user: [last_speak, last_channel]}}
                        global maw_auto_sample
                        maw_auto_sample[message.guild.id] = True
                        try:
                            maw_auto_watched[message.guild.id]
                        except:
                            maw_auto_watched[message.guild.id] = {message.author: [time.time(), message.channel]}
                            print("user added to the auto watched")
                        else:
                            maw_auto_watched[message.guild.id][message.author] = [time.time(), message.channel]
                            print("user added to the auto watched")
                    maw_auto_watched
                    relative_server_path = "./servers/" + str(message.guild.id)
                    relative_thread_path = "./characters/" + str(message.guild.id) + "/" + str(message.channel.id)
                    if os.path.isdir(relative_thread_path):
                        return
                    elif os.path.isdir(relative_server_path):
                        config = read_config(relative_server_path)
                        if not isinstance(message.channel, discord.DMChannel):
                            config.system_prompt = config.system_prompt + message.guild.name + " in channel " + message.channel.name
                        character = MawCharacter("Maw", config, True)
                        if os.path.isfile(relative_server_path + "/history.txt"):
                            history = character.read_history()
                            try:
                                old_message_id = history[-1].message_id
                                if isinstance(message.channel, discord.DMChannel):
                                    old_message_id = int(old_message_id)
                                elif "-" in old_message_id:
                                    old_message_id = old_message_id.split("-")
                                else:
                                    old_message_id = None
                            except:
                                old_message_id = None
                            history.append(MawCharacterMessage(content=str(
                                "The user " + (message.author.global_name if message.author.global_name else message.author.name) + " said: " + message.content), message_id=message.id, role="user"))
                            character.write_history(history)
                            if old_message_id:
                                try:
                                    if isinstance(message.channel, discord.DMChannel):
                                        old_message = await message.channel.fetch_message(old_message_id)
                                        await old_message.edit(view=None)
                                    else:
                                        if old_message_id:
                                            print(int(old_message_id[1]))
                                            channel = await client.fetch_channel(int(old_message_id[1]))
                                            print(channel)
                                            old_message = await channel.fetch_message(int(old_message_id[0]))
                                            await old_message.edit(view=None)
                                except Exception as e:
                                    print(repr(e))
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


@client.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return
    if not after.channel:
        return
    voice_file = open("voice_watch.txt", "r")
    voice_file_lines = voice_file.readlines()
    voice_file.close()
    watched = [int(i.split("-")[0][:-1]) if i[:-1] == "\n" else int(i.split("-")[0]) for i in voice_file_lines]
    if after.channel.id in watched:
        text_channel = None
        for idx, x in enumerate(watched):
            if x == after.channel.id:
                text_channel = int(voice_file_lines[idx].split("-")[1])
        if after.channel.members != []:
            global maw_voice_channels
            if not member.guild in [x.guild for x in maw_voice_channels] and text_channel != None:
                try:
                    proto = await after.channel.connect(cls=VoiceRecvClient)
                    channel = await client.fetch_channel(text_channel)
                    sent_message = await channel.send(
                        "Starting a voice session in " + str(channel.name))
                    thread = await sent_message.create_thread(name="Maw Voice Session")
                    session = MawVoiceSession(guild=member.guild, message=sent_message, thread=thread, proto=proto)
                    await sent_message.edit(view=VoiceTranscribe(session=session))
                    global exclusive
                    exclusive[session] = True
                    maw_voice_channels.append(session)
                    threading.Thread(target=await_voice_allocation, args=[session]).start()
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(repr(e))


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


async def async_await_voice_allocation(session):
    # don't hold up the client thread waiting for allocation
    threading.Thread(target=voice_channel_watcher, args=[session]).start()
    vram.allocate("Maw")
    global stay_allocated
    stay_allocated += 1
    alloc_message = None
    async for i in vram.wait_for_allocation("Maw"):
        if alloc_message == None:
            alloc_message = asyncio.run_coroutine_threadsafe(
                coro=session.thread.send("Waiting for " + str(
                    i) + " before starting\n(Audio spoken during this time will still be transcribed)"),
                loop=client.loop).result()
        else:
            alloc_message = asyncio.run_coroutine_threadsafe(
                coro=alloc_message.edit("Waiting for " + str(
                    i) + " before starting\n(Audio spoken during this time will still be transcribed)"),
                loop=client.loop)
    if alloc_message != None:
        alloc_message = asyncio.run_coroutine_threadsafe(
            coro=alloc_message.edit("Loading models..."),
            loop=client.loop).result()
    # if not transcribe_only:
    # This thread must occur first otherwise whisper freaks out (I think?)
    # Is definitely an issue in MawChat
    # speech_thread = threading.Thread(target=load_speech)
    # speech_thread.start()
    # quick switching + allocation means there's not much reason not to keep the speech model loaded
    speech_thread = threading.Thread(target=load_speech)
    speech_thread.start()
    threading.Thread(target=load_whisper).start()
    global whisperusers
    whisperusers += 1


def await_voice_allocation(session):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(async_await_voice_allocation(session))


@client.slash_command(dm_permission=False)
async def voice(
        interaction: discord.Interaction,
):
    pass


@voice.subcommand(description="Maw for voice channels")
async def join(
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
            try:
                await interaction.guild.change_voice_state(channel=channel, self_mute=False)
            except Exception as e:
                print("Unable to unmute:")
                print(e)
                print(repr(e))
            sent_message = await interaction.response.send_message("Starting a voice session in " + str(channel.name))
            sent_message = await sent_message.fetch()
            thread = await sent_message.create_thread(name="Maw Voice Session")
            session = MawVoiceSession(guild=interaction.guild, message=sent_message, thread=thread, proto=proto)
            if transcribe_only:
                await sent_message.edit(view=VoiceTranscribe(session=session))
            else:
                await sent_message.edit(view=VoiceResponse(session=session))
            global exclusive
            exclusive[session] = transcribe_only
            maw_voice_channels.append(session)
            threading.Thread(target=await_voice_allocation, args=[session]).start()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(repr(e))
            await interaction.response.send_message("Failed to connect to that channel!")
    else:
        await interaction.response.send_message("Already connected to a channel in this server!")


@voice.subcommand(description="Enlist a voice channel to auto-start transcription sessions.", name="enlist")
async def voice_enlist(
        interaction: discord.Interaction,
        channel: discord.VoiceChannel,
):
    if not isinstance(channel, discord.Thread):
        try:
            voice_file = open("voice_watch.txt", "r")
            voice_file_lines = voice_file.readlines()
            voice_file.close()
            for voice_channel in voice_file_lines:
                if voice_channel.split("-")[0] == str(channel.id):
                    await interaction.response.send_message("This channel is already enlisted for auto-transcriptions!")
                    return
            voice_file = open("voice_watch.txt", "a")
            voice_file.write(str(channel.id) + "-" + str(interaction.channel.id) + "\n")
            voice_file.close()
            await interaction.response.send_message("Enlisted this channel to auto-transcriptions.")
        except:
            await interaction.response.send_message("Failed to enlist!")
    else:
        await interaction.response.send_message("Cannot enlist a thread!")


@voice.subcommand(description="Delist a voice channel from automatic transcriptions", name="delist")
async def voice_delist(
        interaction: discord.Interaction,
        channel: discord.VoiceChannel,
):
    try:
        voice_file = open("voice_watch.txt", "r")
        voice_file_lines = voice_file.readlines()
        voice_file.close()
        channel_found = False
        for voice_channel in voice_file_lines:
            if voice_channel.split("-")[0] == str(channel.id):
                channel_found = True
                break
        if channel_found:
            voice_file = open("voice_watch.txt", "w")
            for line in voice_file_lines:
                if line.split("-")[0] != str(channel.id):
                    voice_file.write(line)
            voice_file.close()
            await interaction.response.send_message("Delisted this channel.")
        else:
            await interaction.response.send_message("This channel is not enlisted!")
    except:
        await interaction.response.send_message("Failed to delist!")


def update_watched_channels():
    global watched_channels
    channel_file = open("channel_watch.txt", "r")
    channel_file_lines = channel_file.readlines()
    channel_list = []
    for each_channel in channel_file_lines:
        channel_list.append(int(each_channel if each_channel[-1] != "\n" else each_channel[:-1]))
    watched_channels = channel_list


@client.slash_command(dm_permission=False, name="channel")
async def channel_command(
        interaction: discord.Interaction,
):
    pass


@channel_command.subcommand(description="Enlist a channel to auto-history.", name="enlist")
async def channel_enlist(
        interaction: discord.Interaction,
):
    try:
        channel = interaction.channel
        channel_file = open("channel_watch.txt", "r")
        channel_file_lines = channel_file.readlines()
        channel_file.close()
        for each_channel in channel_file_lines:
            if each_channel.split("-")[0] == str(channel.id):
                await interaction.response.send_message("This channel is already enlisted for auto-history!")
                return
        channel_file = open("channel_watch.txt", "a")
        channel_file.write(str(interaction.channel.id) + "\n")
        channel_file.close()
        await interaction.response.send_message("Enlisted this channel to auto-history.")
        update_watched_channels()
    except:
        await interaction.response.send_message("Failed to enlist!")


@channel_command.subcommand(description="Enlist a channel to auto-history.", name="enlist_all")
async def channel_enlist_all(
        interaction: discord.Interaction,
):
    channel_file = open("channel_watch.txt", "r")
    channel_file_lines = channel_file.readlines()
    channel_file.close()
    for channel in [x for x in interaction.guild.channels if isinstance(x, discord.TextChannel)]:
        try:
            enlisted = False
            for each_channel in channel_file_lines:
                if each_channel.split("-")[0] == str(channel.id):
                    await interaction.response.send_message("This channel is already enlisted for auto-history!")
                    enlisted = True
            if not enlisted:
                channel_file = open("channel_watch.txt", "a")
                channel_file.write(str(channel.id) + "\n")
                channel_file.close()
        except:
            await interaction.response.send_message("Failed to enlist!")
    update_watched_channels()
    await interaction.response.send_message("Enlisted all channels to auto-history.")


@channel_command.subcommand(description="Delist a channel from auto-history.", name="delist")
async def channel_delist(
        interaction: discord.Interaction,
):
    try:
        channel = interaction.channel
        channel_file = open("channel_watch.txt", "r")
        channel_file_lines = channel_file.readlines()
        channel_file.close()
        channel_found = False
        for each_channel in channel_file_lines:
            if each_channel.split("-")[0] == str(channel.id):
                channel_found = True
                break
        if channel_found:
            channel_file = open("voice_watch.txt", "w")
            for line in channel_file_lines:
                if (line if line[-1] != "\n" else line[:-1]) != str(channel.id):
                    channel_file.write(line)
            channel_file.close()
            await interaction.response.send_message("Delisted this channel.")
            update_watched_channels()
        else:
            await interaction.response.send_message("This channel is not enlisted!")
    except:
        await interaction.response.send_message("Failed to delist!")


@client.slash_command(dm_permission=True, name="token")
async def token(
        interaction: discord.Interaction,
):
    pass


@token.subcommand(description="Get the current token count of any conversation.", name="count")
async def token_count(
        interaction: discord.Interaction,
):
    relative_server_path = "./servers/" + str(interaction.guild.id)
    relative_thread_path = "./characters/" + str(interaction.guild.id) + "/" + str(interaction.channel.id)
    if os.path.isdir(relative_thread_path):
        config = read_config("./characters/" + str(interaction.guild.id) + "/" + str(interaction.channel.id))
        character = MawCharacter(config.name, config, False)
        history = character.read_history()
        model_input, local_token_count = history_to_llama(history, config, "", math.pow(10, 100), character.maw)
        await interaction.response.send_message(str(local_token_count) + " tokens.", ephemeral=True)
    elif os.path.isdir(relative_server_path):
        config = read_config("./servers/" + str(interaction.guild.id))
        character = MawCharacter("Maw", config, True)
        history = character.read_history()
        model_input, local_token_count = history_to_llama(history, config, "", math.pow(10, 100), character.maw)
        await interaction.response.send_message(str(local_token_count) + " tokens.", ephemeral=False)
    else:
        await interaction.response.send_message("0 tokens.", ephemeral=False)


update_watched_channels()
threading.Thread(target=watcher).start()
threading.Thread(target=auto_watcher).start()
# threading.Thread(target=model_factory).start()
threading.Thread(target=typing_watcher).start()
client.run(TOKEN)
