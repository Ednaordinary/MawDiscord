# Maw

## Talk with a local llm with all your friends, straight in discord



https://github.com/user-attachments/assets/8869e6d1-02b9-4ce3-aa8d-c10e65f7dcb5

> [!NOTE]
> At one point, Maw could even talk back and forth with you in a voice channel! This feature proved unsustainable but I plan to implement it in a future bot. Stay tuned!

## Or.. make your own characters! Edit anything you want, and make it your own!



https://github.com/user-attachments/assets/04419b07-d64b-40af-88ac-ab91e10b88cf


## Maw can even generate images for you!

[Dante](https://github.com/Ednaordinary/DanteMode) must be installed alongside Maw, cloned to the same directory

![image](https://github.com/user-attachments/assets/faba6cf5-7f33-4df8-a8b8-7203acf19150)



## Requirements

- Linux with cuda drivers installed
- NVIDIA gpu with 24GB vram
- A decent amount of storage
- Python3.12+ (3.13 freethreaded version highly encouraged for the best performance. 3.12 is the earliest tested version but earlier may also work)
- uv

## Install

```sh
git clone https://github.com/Ednaordinary/MawDiscord/
cd MawDiscord
pip install uv
uv venv -p 3.13t
source .venv/bin/activate
uv pip install -r pyproject.toml
```

Merge the following PRs into your environment:

- https://github.com/huggingface/transformers/pull/31301/
- https://github.com/nextcord/nextcord/pull/1160

Place your bot token as `DISCORD_TOKEN` in a .env file

## Run

```sh
source .venv/bin/activate # if you haven't already activated the environment
PYTHON_GIL=0 python3 ./main.py
```
