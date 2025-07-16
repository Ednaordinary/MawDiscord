import re

class Tool:
    doc = "You are able to use 'tools', a set of functions that allows you to access external functionality. Please refer to the documentation for each tool and its use and follow it exactly. All tools must be called outside of your <think> and </think> tags, otherwise they will not trigger."
    def __init__(self):
        pass
    def filter(self, answer):
        return answer
    def will_run(self, answer):
        return False
    async def run(self, answer):
        return

class DanteTool(Tool):
    doc = "You are able to generate or create images by asking another assistant known as Dante: simply enclose a description of an image in <- and ->, for example: <-A hot dog on a grill, the grill sits on a wooden table with condiments on it->. If asked to make an image, go ahead and generate it"
    def __init__(self, get_hook, dante_id, check_perm, channel, hooks, loop, client_id):
        super().__init__()
        self.regex = re.compile(r'<-[\S\s]+?>')
        self.get_hook = get_hook
        self.dante_id = dante_id
        self.check_perm = check_perm
        self.channel = channel
        self.hooks = hooks
        self.loop = loop
        self.client_id = client_id
    def filter(self, answer):
        return re.sub(self.regex, "", answer)
    def will_run(self, answer):
        for i in re.findall(self.regex, answer):
            return True
        return False
    async def run(self, answer):
        try:
            members = self.channel.members
            hook = True
            for i in members:
                if i.id == self.dante_id:
                    if self.check_perm(self.channel, self.channel.guild, "send"):
                        hook = False
            if hook:
                print("getting hook")
                hook = await self.get_hook(self.channel, self.hooks, self.client_id)
                print("got hook")
                if hook == None:
                    return
            for image in re.findall(self.regex, answer):
                image = image[2:-1]
                if image[-1] == "-": image = image[:-1]
                if image.strip() != "":
                    try:
                        print("writing to queue")
                        with open("../DanteMode/queue.txt", "a") as image_queue:
                            if hook:
                                image_queue.write("\n" + str(self.channel.id) + "|" + hook.url + "|" + str(image).replace("\n", "\\n"))
                            else:
                                image_queue.write("\n" + str(self.channel.id) + "||" + str(image).replace("\n", "\\n"))
                    except Exception as e:
                        print("Dante's queue could not be written to. Please check it is installed.")
                        print(repr(e))
                        print(e)
        except Exception as e:
            print("Error in tool runner")
            print(repr(e))
            print(e)

class SynapseTool(Tool):
    doc = "You are able to ask another assistant known as Synapse to search the web for the answer to a question. Simply enclose a prompt in $% and %$, like this: $%Who is Edna Mode?%$. Please only use this tool if you cannot answer the question or the user has requested it."
    def __init__(self, requests):
        super().__init__()
        self.regex = re.compile(r'<-[\S\s]+?>')
        self.requests = requests
    def filter(self, answer):
        return re.sub(r'<-[\S\s]+?>', "", answer)
    def will_run(self, answer):
        if self.regex.search(answer):
            return True
        else:
            return False
    async def run(self, answer):
        for image in re.findall(self.regex, answer):
            pass
