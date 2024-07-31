from __future__ import annotations

import asyncio
import logging
import time

import nextcord

import decrypter

__all__ = [
    'VoiceRecvClient',
]

log = logging.getLogger(__name__)


class VoiceRecvClient(nextcord.VoiceClient):
    def __init__(self, client: nextcord.Client, channel: nextcord.abc.Connectable):
        super().__init__(client, channel)

    async def listen(self):
        """Just yields received data. It's your choice what to do after that :). Doesn't stop until the client disconnects."""
        # TODO: more info

        while self.is_connected():
            yield self.socket.recv(4096)

    def decrypt(self, header, data):
        return getattr(decrypter, f"decrypt_{self.mode}")(self.secret_key, header, data)

    # def _wait_for_user_id(self, ssrc: int) -> int:
    #     ssrc_cache = self.ws
    #     while not (user_data := ssrc_cache.get(ssrc)):
    #         time.sleep(0.01)
    #
    #     return user_data["user_id"]