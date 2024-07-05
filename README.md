# Maw

Discord implementation of Maw. Also features character creation with /character

## Requirements

- Linux with cuda drivers installed
- NVIDIA gpu with 24GB vram
- A decent amount of storage
- Python3.12 and the venv package

## Install

`git clone https://github.com/Ednaordinary/MawDiscord/`

`cd MawDiscord`

`mkdir character`

`mkdir servers`

`python3 -m venv venv`

`cd venv`

`source bin/activate`

`cd ..`

`pip install -r requirements.txt`

Merge the following PRs into your environment:

- https://github.com/huggingface/transformers/pull/31301/
- https://github.com/nextcord/nextcord/pull/1160

Place your bot token as `DISCORD_TOKEN` in a .env file

## Run

`cd venv`

`source bin/activate` if you haven't already activated the environment

`cd ..`

`python3 ./main.py`
