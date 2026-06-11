

class Neuron:
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.content = ""
        self.tokens = 0

    def update_answer(self, idx, update, tok_count, limit=True):
        # most args do not matter and are to match ScrollRedoView.update_answer
        self.content = update
        self.tokens = tok_count

    def complete_answer(self, idx):
        pass

class Cortex:
    
