# Maw

## Talk with a local llm with all your friends, straight in discord



https://github.com/user-attachments/assets/8869e6d1-02b9-4ce3-aa8d-c10e65f7dcb5


## Or.. make your own characters! Edit anything you want, and make it your own!



https://github.com/user-attachments/assets/04419b07-d64b-40af-88ac-ab91e10b88cf


## Want to voice chat with it? Transcribe discord calls?



https://github.com/user-attachments/assets/7317a1e6-a4c1-4cc9-99bf-4b768f4bdc20


## Maw can even generate images for you!

[Dante](https://github.com/Ednaordinary/DanteMode) must be installed alongside Maw, cloned to the same directory

![image](https://github.com/user-attachments/assets/faba6cf5-7f33-4df8-a8b8-7203acf19150)



## Requirements

- Linux with cuda drivers installed
- NVIDIA gpu with 24GB vram
- A decent amount of storage
- Python3.12 and the venv package

## Install

```sh
git clone https://github.com/Ednaordinary/MawDiscord/
cd MawDiscord
mkdir characters
mkdir servers
python3 -m venv venv
cd venv
source bin/activate
cd ..
pip install -r requirements.txt
```

Merge the following PRs into your environment:

- https://github.com/huggingface/transformers/pull/31301/
- https://github.com/nextcord/nextcord/pull/1160

Place your bot token as `DISCORD_TOKEN` in a .env file

## Run

```sh
cd venv
source bin/activate # if you haven't already activated the environment
cd ..
python3 ./main.py
```
