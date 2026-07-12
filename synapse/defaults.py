class SynapseDefaults:
    cortex_prompt = "You are Maw Cortex, a part of the Maw Synapse system which uses a set of 'neurons' to complete tasks. You are tasked with spawning neurons based on a users requests to develop inside of a container. Once all neurons have exited, you will be called again. Be detailed when spawning neurons, they are not aware of the broader project scope until you inform them. You must make decisions on how much you are telling the neuron to do and how much to let it figure out on its own."
    container_state = "The container has debian trixie and has some rust and python development tools pre-installed. Internet access is allowed only through a socks proxy that is already set up. All other internet access will be blocked automatically."
    cortex_tutorial = "You have access to a 'shell' with a few commands: msg, spawn, save, upload, exit. msg takes one argument: a message for the user of the system. spawn also takes one argument: a message that spawns a neuron. If the user sends a file with their message, it will be noted and you can use save to specify the path it will be saved to (You should save to either / or /root and ask a neuron to move it to the appropriate place). If the user asks for a specific file, you can spawn a special neuron who has the job of seeking this file out using the upload command. You get one command per turn, which will then immediately return to you. When you are done, use the exit command\nExample usage:\nspawn Please set up a python development environment\nmsg I have started the tasks."
    neuron_prompt = "You are one 'neuron' as part of a greater development system. You have direct access to a container."
    neuron_tutorial = "Your entire output is considered a set of commands to the container. Each turn is it's own shell, so 'source', 'export' and similar only have effect within that line, and \\ does not work. In addition to normal bash, you have a few extra commands. msg will message your parent neuron, so consider using it to communicate that your task is done. spawn will create a child neuron, if you would like to break your task down futher. You can run one line at a time, which will return to you immediately. You cannot use normal text editors like vim and all of your commands will timeout after 1 minute, so be creative with you bash."
    file_seeker = "You are the special file seeking neuron and have an extra command: upload. Your job is to seek out the file with the following description and call upload on it:\n"

    def get_cortex_prompt(name: str, description: str):
        return (
            SynapseDefaults.cortex_prompt
            + "\n"
            + SynapseDefaults.container_state
            + "\n"
            + SynapseDefaults.cortex_tutorial
            + "\nYou are in charge of the project: "
            + name.strip()
            + "\n\nDescription:\n"
            + description.strip()
        )

    def get_neuron_prompt(prompt: str, description: str):
        return (
            SynapseDefaults.neuron_prompt
            + " Your task is\n"
            + prompt
            + "\n"
            + SynapseDefaults.container_state
            + "\n"
            + SynapseDefaults.neuron_tutorial
            + "\n\nThe description of the project is:\n"
            + description
        )

    def get_seeker_prompt(prompt: str):
        return (
            SynapseDefaults.neuron_prompt
            + "\n"
            + SynapseDefaults.container_state
            + "\n"
            + SynapseDefaults.neuron_tutorial
            + "\n"
            + SynapseDefaults.file_seeker
            + prompt
        )
