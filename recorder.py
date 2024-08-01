from __future__ import annotations

import asyncio
import logging
import time
from select import select
from socket import socket

import nextcord
import decrypter
from typing import Final
import speech_recognition as sr
import array
import audioop

__all__ = [
    'VoiceRecvClient',
]

log = logging.getLogger(__name__)

# this file takes HEAVY inspiration from both https://github.com/imayhaveborkedit/discord-ext-voice-recv/ and https://github.com/nextcord/nextcord/pull/1113

class VoiceRecvClient(nextcord.VoiceClient):
    def __init__(self, client: nextcord.Client, channel: nextcord.abc.Connectable):
        super().__init__(client, channel)

    async def connect(self, *, reconnect: bool, timeout: float) -> None:
        await super().connect(reconnect=reconnect, timeout=timeout)
        await self.ws.loop.sock_connect(self.socket, (self.endpoint_ip, self.voice_port))

    async def listen(self):
        """Just yields received data. It's your choice what to do after that :). Doesn't stop until the client disconnects."""
        while self.is_connected():
            try:
                ready, _, _ = select([self.socket], [], [], 30)
            except Exception as e:
                print(repr(e))
                continue
            while not ready:
                print("Unable to read")
                time.sleep(0.01)
            try:
                data = self.socket.recv(1024) # lower means faster latency probably
            except Exception as e:
                print("Exception while receiving data:")
                print(repr(e))
            else:
                yield data

    def decrypt(self, header, data):
        return getattr(decrypter, f"decrypt_{self.mode}")(self.secret_key, header, data)

    def _wait_for_user_id(self, ssrc: int) -> int:
        ssrc_cache = self.ws.ssrc_cache
        while not (user_data := ssrc_cache.get(ssrc)):
            time.sleep(0.01)

        return user_data["user_id"]

# class entirely from https://github.com/imayhaveborkedit/discord-ext-voice-recv/blob/main/discord/ext/voice_recv/extras/speechrecognition.py
class BytesSRAudioSource(sr.AudioSource):
    little_endian: Final[bool] = True
    SAMPLE_RATE: Final[int] = 48_000
    SAMPLE_WIDTH: Final[int] = 2
    CHANNELS: Final[int] = 2
    CHUNK: Final[int] = 960

    def __init__(self, buffer: array.array[int]):
        self.buffer = buffer
        self._entered: bool = False

    @property
    def stream(self):
        return self

    def __enter__(self):
        if self._entered:
            log.warning('Already entered sr audio source')
        self._entered = True
        return self

    def __exit__(self, *exc) -> None:
        self._entered = False
        if any(exc):
            log.exception('Error closing sr audio source')

    def read(self, size: int) -> bytes:
        for _ in range(10):
            if len(self.buffer) < size * self.CHANNELS:
                time.sleep(0.1)
            else:
                break
        else:
            if len(self.buffer) == 0:
                return b''

        chunksize = size * self.CHANNELS
        audiochunk = self.buffer[:chunksize].tobytes()
        del self.buffer[: min(chunksize, len(audiochunk))]
        audiochunk = audioop.tomono(audiochunk, 2, 1, 1)
        return audiochunk

    def close(self) -> None:
        self.buffer.clear()