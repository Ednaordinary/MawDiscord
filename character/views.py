import nextcord as discord
from .history import Message
import asyncio
import time
import re

verbose = True

think_regex = re.compile(r'.*?<\/think>', flags=re.DOTALL)

class EditModal(discord.ui.Modal):
    def __init__(self, original_content, message_id, role, history):
        super().__init__(title="Edit Message", timeout = 60 * 60 * 24)
        self.original_content = original_content
        self.message_id = message_id
        self.role = role
        self.history = history
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
        self.history.edit_message(message)
        await interaction.response.edit_message(content=self.content.value)

class ScrollRedoView(discord.ui.View):
    def __init__(self, answers, history, user_message, message, prompt, tools, idx=0, edit=True, queue=None, loop=None, timeout = None, cutoff=None, continue_request=None):
        super().__init__(timeout=timeout)
        self.answers = answers
        self.history = history
        self.user_message = user_message
        self.message = message
        self.prompt = prompt
        self.tools = tools
        self.completed = [False]*len(answers)
        self.idx = idx
        self.edit = edit
        self.limiter = time.perf_counter()
        self.runtools = self.get_runtools()
        self.menu = self.get_menu()
        self.thought_button = self.get_thought_button()
        self.show_menu = False
        self.handle_disabled()
        self.queue = queue
        self.loop = loop
        self.cutoff = cutoff
        self.continue_request = continue_request
    def get_idx(self):
        return self.idx
    def set_idx(self):
        self.idx = idx
    def update_answer(self, idx, updated, limit=True):
        self.answers[idx] = updated
        if idx == self.idx:
            if self.limiter + 1.1 < time.perf_counter() or limit == False:
                self.history.edit_message(Message(self.message.id, updated, "assistant"))
                answer = self.get_answer()
                if verbose: print("Editing for", idx, "|", answer[:10])
                self.handle_disabled()
                asyncio.run_coroutine_threadsafe(self.message.edit(content=answer[:1999], view=self), self.loop)
                self.limiter = time.perf_counter()
    def complete_answer(self, idx):
        self.completed[idx] = True
        self.handle_disabled()
        answer = self.get_answer()
        if self.idx == idx:
            asyncio.run_coroutine_threadsafe(self.message.edit(content=answer[:1999], view=self), self.loop)
        elif self.completed[self.idx]:
            asyncio.run_coroutine_threadsafe(self.message.edit(view=self), self.loop)
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
            scroll_request = self.continue_request(self.user_message, self.message, self.prompt, self.cutoff, self.tools, self.edit, self, idxs)
            self.queue.put(scroll_request)
        if self.menu not in self.children:
            self.children.append(self.menu)
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
                if self.edit == False or not self.menu:
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
        self.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "", view=self)
    @discord.ui.button(label="1", style=discord.ButtonStyle.grey)
    async def number(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.show_menu = not self.show_menu 
        self.handle_disabled()
        await interaction.response.edit_message(view=self)
    @discord.ui.button(row=3, label="🖊️", style=discord.ButtonStyle.primary)
    async def edit_message(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_modal(EditModal(self.answers[self.idx], interaction.message.id, "assistant", self.history))
    @discord.ui.button(label="➡️", style=discord.ButtonStyle.primary)
    async def scroll_forward(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.idx += 1
        self.handle_disabled()
        answer = self.get_answer()
        message = Message(interaction.message.id, answer, "assistant")
        self.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "", view=self)
    @discord.ui.button(label="▶️", style=discord.ButtonStyle.green)
    async def run_tools(self, button: discord.ui.Button, interaction: discord.Interaction):
        answer = self.get_answer(do_filter=False)
        runnabletools = [i for i in self.tools if i.will_run(answer)]
        for tool in runnabletools:
            tool.run(answer)
    @discord.ui.string_select(row=2, min_values=1, max_values=1, options=[])
    async def select_prompt(self, select: discord.ui.StringSelect, interaction: discord.Interaction):
        self.idx = int(select.values[0])
        self.handle_disabled()
        answer = self.get_answer()
        message = Message(interaction.message.id, answer, "assistant")
        self.history.edit_message(message)
        await interaction.response.edit_message(content=answer[:1999] if answer != "" else "", view=self)
    @discord.ui.button(row=3, label="💭", style=discord.ButtonStyle.grey)
    async def send_thoughts(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message(content=self.get_thoughts()[:1999], ephemeral=True)
