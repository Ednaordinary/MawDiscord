from __future__ import annotations

import asyncio
import logging
import time
from select import select
from socket import socket

import nextcord

import decrypter

__all__ = [
    'VoiceRecvClient',
]

log = logging.getLogger(__name__)


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
            print("Socket data returned")
            try:
                data = self.socket.recv(4096)
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