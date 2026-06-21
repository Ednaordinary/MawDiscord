import sys
import discord
import traceback

from .defaults import SynapseDefaults
from .container import Container

sys.path.append("..")
from character.history import Message
from character.config import Config
from util import get_path, get_history, FakeHistObj


class SynapseModal(discord.ui.Modal):
    def __init__(self, histories):
        super().__init__(
            title="Create Synapse Session",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.histories = histories
        self.name = discord.ui.TextInput(
            label="Project Name",
            style=discord.TextStyle.short,
            placeholder="Name of the synapse session",
            required=True,
            min_length=0,
            max_length=50,
        )
        self.add_item(self.name)

        self.description = discord.ui.TextInput(
            label="Description",
            style=discord.TextStyle.paragraph,
            placeholder="Describe the project with whatever details synapse needs to complete it",
            required=True,
            min_length=30,
            max_length=4000,
        )
        self.add_item(self.description)

        self.cortex_prompt = discord.ui.TextInput(
            label="Cortex Prompt",
            placeholder="Additional prompt for Cortex",
            style=discord.TextStyle.paragraph,
            required=False,
            min_length=10,
            max_length=1000,
        )

        self.add_item(self.cortex_prompt)

        self.neuron_prompt = discord.ui.TextInput(
            label="Neuron Prompt",
            placeholder="Additional prompt for Neurons",
            style=discord.TextStyle.paragraph,
            required=False,
            min_length=10,
            max_length=1000,
        )

        self.add_item(self.neuron_prompt)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        try:
            root = await interaction.response.send("Starting a synapse session.")
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
                hist_obj = FakeHistObj(thread.id, interaction.guild.id)
                history_path = get_path("syn", "history", hist_obj)
                prompt = SynapseDefaults.get_cortex_prompt(
                    self.name.value, self.description.value
                )
                history = get_history(history_path, self.histories, prompt, char=True)
                config_path = get_path("syn", "config", hist_obj)
                config_file = Config(config_path)
                config = config_file.get()
                config["name"] = self.name.value
                config["description"] = self.description.value
                config["cortex_prompt"] = self.cortex_prompt.value or ""
                config["neuron_prompt"] = self.neuron_prompt.value or ""
                container = Container()
                config["container"] = container.name
                config_file.write(config)
        except:
            print(traceback.print_exc())
