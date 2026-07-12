from .container import Container
from .defaults import SynapseDefaults

class Synapse:
    def __init__(self, container: Container, cortex):
        self.container = container
        self.cortex = cortex

class Neuron:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.content = ""
        self.tokens = 0

    def update_answer(self, update, tok_count):
        self.content = update
        self.tokens = tok_count

    def complete_answer(self, idx):
        pass

class Cortex:
    def __init__(self):
        self.children = []
        self.tokens = 0
        self.commands = CortexCommands()

    def update_answer(self, update, tok_count):
        self

class CortexCommands:
    def __init__(self):
        self.cmds = ["msg", "spawn"] # to impl: save, upload
    def msg(self, content, send_hook, syn: Synapse, neuron: Neuron):
        send_hook.send(content)
    def spawn(self, content, send_hook, syn: Synapse, neuron: Neuron):
        
