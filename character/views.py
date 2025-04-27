import nextcord as discord
from .history import Message
from .config import Config
import asyncio
import time
import re
import math

verbose = True

think_regex = re.compile(r'.*?<\/think>', flags=re.DOTALL)

class ScrollRedoView(discord.ui.View):
    def __init__(self, **kwargs):
        # answers, context, tools, idx, edit, queue, loop, timeout, cutoff, continue_request
        self.__dict__.update(kwargs)
    def get_idx(self):
        return self.idx
    def set_idx(self):
        self.idx = idx
    def update_answer(self, idx, updated, limit=True):
        self.answers[idx] = updated
        if idx == self.idx:
            limiter = math.log(len(set(self.context.history.workers)) + 3)
            if self.limiter + limiter < time.perf_counter() or limit == False:
                self.context.history.edit_message(Message(self.context.bot_message.id, updated, "assistant"))
                answer = self.get_answer()
                if verbose: print("Editing for", idx, "|", answer[:10])
                self.handle_disabled()
                try:
                    asyncio.run_coroutine_threadsafe(self.context.bot_message.edit(content=answer[:1999], view=self), self.loop).result()
                except Exception as e:
                    print(e)
                    print(repr(e))
                self.limiter = time.perf_counter()
    def complete_answer(self, idx):
        self.completed[idx] = True
        self.handle_disabled()
        answer = self.get_answer()
        if self.idx == idx:
            asyncio.run_coroutine_threadsafe(self.context.bot_message.edit(content=answer[:1999], view=self), self.loop)
        elif self.completed[self.idx]:
            asyncio.run_coroutine_threadsafe(self.context.bot_message.edit(view=self), self.loop)
    def get_runtools(self):
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.label == "▶️":
                return child
    def get_menu(self):
        for child in self.children:
            if isinstance(child, discord.ui.StringSelect):
                return child
    def get_edit_button(self):
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.label == "🖊️":
                return child
    def get_thought_button(self):
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.label == "💭":
                return child
    def handle_disabled(self):
        if self.idx == len(self.answers) - 1:
            idxs = [x + len(self.answers) for x in range(5)]
            self.answers.extend([""]*5)
            self.completed.extend([False]*5)
            scroll_request = self.continue_request(self.context.message, self.context.bot_message, self.context.prompt, self.cutoff, self.tools, self.edit, self, idxs)
            self.queue.put(scroll_request)
        if self.menu not in self.children:
            self.children.append(self.menu)
        if self.edit_button not in self.children:
            self.children.append(self.edit_button)
        if self.thought_button not in self.children:
            self.children.append(self.thought_button)
        if self.run_tools not in self.children:
            self.children.append(self.run_tools)
        for child in self.children: 
            if isinstance(child, discord.ui.StringSelect):
                if self.show_menu == True:
                    child.options = [discord.SelectOption(label=(self.get_answer(idx)[:99] if self.get_answer(idx).strip() != "" else "..."), value=str(idx), default=(idx==self.idx)) for idx in range(len(self.answers))][:25]
                else:
                    if self.menu in self.children:
                        self.children = [x for x in self.children if x != self.menu]
                continue
            elif child.label == "⬅️":
                if self.idx == 0:
                    child.disabled = True
                else:
                    child.disabled = False
            elif child.label == "🖊️":
                if self.edit == False or not self.show_menu:
                    self.children = [x for x in self.children if x != child]
                elif self.completed[self.idx]:
                    child.disabled = False
                else:
                    child.disabled = True
            elif child.label == "💭":
                if self.get_thoughts() == "" or self.show_menu == False:
                    self.children = [x for x in self.children if x != self.thought_button]
            elif child.label == "▶️":
                answer = self.get_answer(do_filter=False)
                runnabletools = [i for i in self.tools if i.will_run(answer)]
                if runnabletools == []:
                    self.children = [x for x in self.children if x != self.runtools]
            else:
                try:
                    int(child.label)
                except:
                    pass
                else:
                    child.label = str(self.idx + 1)
    def get_answer(self, idx=None, do_filter=True):
        if idx == None:
            idx = self.idx
        answer = self.answers[idx]
        for match in re.findall(think_regex, answer):
            answer = answer.replace(match, "").strip()
        if idx == self.idx and do_filter:
            for tool in self.tools:
                answer = tool.filter(answer)
        return answer
    def get_thoughts(self):
        answer = self.answers[self.idx]
        thoughts = []
        for match in re.findall(think_regex, answer):
            thoughts.append(match.replace("</think>", ""))
        return "\n".join(thoughts)
    @discord.ui.button(label="⬅️", style=discord.ButtonStyle.primary)
    async def scroll_back(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.idx -= 1
        self.handle_disabled()
        answer = self.get_answer()
        message = Message(interaction.message.id, answer, "assistant")
        self.context.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "", view=self)
    @discord.ui.button(label="1", style=discord.ButtonStyle.grey)
    async def number(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.show_menu = not self.show_menu 
        self.handle_disabled()
        await interaction.response.edit_message(view=self)
    @discord.ui.button(row=3, label="🖊️", style=discord.ButtonStyle.primary)
    async def edit_message(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_modal(EditModal(self.answers[self.idx], interaction.message.id, "assistant", self.context.history))
    @discord.ui.button(label="➡️", style=discord.ButtonStyle.primary)
    async def scroll_forward(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.idx += 1
        self.handle_disabled()
        answer = self.get_answer()
        message = Message(interaction.message.id, answer, "assistant")
        self.context.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "...", view=self)
    @discord.ui.button(label="▶️", style=discord.ButtonStyle.green)
    async def run_tools(self, button: discord.ui.Button, interaction: discord.Interaction):
        answer = self.get_answer(do_filter=False)
        await interaction.response.edit_message(view=self)
        runnabletools = [i for i in self.tools if i.will_run(answer)]
        for tool in runnabletools:
            try:
                await tool.run(answer)
            except Exception as e:
                print(repr(e))
    @discord.ui.string_select(row=2, min_values=1, max_values=1, options=[])
    async def select_prompt(self, select: discord.ui.StringSelect, interaction: discord.Interaction):
        self.idx = int(select.values[0])
        self.handle_disabled()
        answer = self.get_answer()
        message = Message(interaction.message.id, answer, "assistant")
        self.context.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "...", view=self)
    @discord.ui.button(row=3, label="💭", style=discord.ButtonStyle.grey)
    async def send_thoughts(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message(content=self.get_thoughts()[:1999], ephemeral=True)

class EditModal(discord.ui.Modal):
    def __init__(self, original_content, message_id, role, history):
        super().__init__(title="Edit Message", timeout = 60 * 60 * 24)
        self.original_content = original_content
        self.message_id = message_id
        self.role = role
        self.context.history = history
        self.content = discord.ui.TextInput(
            label="Message",
            style=discord.TextInputStyle.paragraph,
            placeholder="New content of the message",
            default_value=self.original_content[:1999],
            required=True,
            min_length=1,
            max_length=1999,
        )
        self.add_item(self.content)
        
    async def callback(self, interaction: discord.Interaction):
        message = Message(self.message_id, self.content.value, self.role)
        self.context.history.edit_message(message)
        await interaction.response.edit_message(content=self.content.value)

class EditButton(discord.ui.View):
    def __init__(self, history, role, message_id=None):
        super().__init__(timeout=None)
        self.context.history = history
        self.role = role
        self.message_id = message_id

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.blurple)
    async def edit_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        edit_modal = EditModal(interaction.message.content, (self.message_id if self.message_id != None else interaction.message.id), self.role, self.context.history)
        await interaction.response.send_modal(edit_modal)

class ResetContextButton(discord.ui.View):
    def __init__(self, *, timeout=None, history):
        super().__init__(timeout=timeout)
        self.context.history = history

    @discord.ui.button(label="Yes", style=discord.ButtonStyle.red)
    async def reset_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        ignored_ids = [0]
        if len([x for x in self.context.history.history if x not in ignored_ids]) > 0:
            await interaction.response.edit_message(content="Context deleted.", view=None)
            self.context.history.touch_history()
        else:
            await interaction.response.edit_message(content="No context found to delete.", view=None)

class CharacterModal(discord.ui.Modal):
    def __init__(self, histories, get_path, get_history):
        super().__init__(
            title="Make Character",
            timeout=60 * 60 * 24,  # 1 day
        )
        self.histories = histories
        self.get_path = get_path
        self.get_history = get_history
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
            max_length=1900,
        )
        self.add_item(self.description)

        self.environment = discord.ui.TextInput(
            label="Starting environment",
            placeholder="The environment the character starts in. Ex: You in a large mansion.",
            style=discord.TextInputStyle.paragraph,
            required=True,
            min_length=10,
            max_length=1000,
        )

        self.add_item(self.environment)

    async def callback(self, interaction: discord.Interaction) -> None:
        if self.description.value[-1] != ".":
            description = self.description.value + "."
        else:
            description = self.description.value
        prompt = "Your name is " + self.name.value + ". " + description
        root = await interaction.send(prompt)
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
            history_path = self.get_path("char", "history", char_id=thread.id, server_id=interaction.guild.id)
            history = self.get_history(history_path, self.histories, prompt)
            config_path = self.get_path("char", "config", char_id=thread.id, server_id=interaction.guild.id)
            config = Config(config_path)
            config.write(self.name.value)
            await root.edit(view=EditButton(history, "system", message_id=0))
            env_message = await thread.send(self.environment.value, view=EditButton(history, "system"))
            history.add_message(Message(env_message.id, env_message.content, "system"))

class HookMessage():
    def __init__(self, id, hook, thread):
        self.id = id
        self.hook = hook
        self.thread = thread
    async def edit(self, **kwargs):
        await self.hook.edit_message(**kwargs, message_id=self.id, thread=self.thread)
