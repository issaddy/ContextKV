"""Request-local prompt measurements; no reuse of Native cache observations."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256

from .serialization import dumps


class TurnTemplateMemo:
    """Keep small prompt objects and many cheap exact length measurements.

    The template/fingerprint is immutable for this turn. A length is reusable
    only for identical messages and offset mode. Future quotes and KV hits
    are deliberately not cached here.
    """

    def __init__(self, template, *, max_prompts=32, max_lengths=4096):
        self._template = template
        self._prompts = OrderedDict()
        self._lengths = OrderedDict()
        self._max_prompts, self._max_lengths = max_prompts, max_lengths

    def __getattr__(self, name):
        return getattr(self._template, name)

    @staticmethod
    def _key(messages, with_offsets=False):
        return sha256(dumps((messages, with_offsets)).encode()).digest()

    @staticmethod
    def _put(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def render(self, messages, *, with_offsets=False):
        return self.render_many((messages,), with_offsets=with_offsets)[0]

    def render_many(self, conversations, *, with_offsets=False):
        keys = tuple(self._key(messages, with_offsets) for messages in conversations)
        missing = {key: messages for key, messages in zip(keys, conversations)
                   if key not in self._prompts}
        rendered = dict(zip(missing, self._template.render_many(tuple(missing.values()),
            with_offsets=with_offsets), strict=True)) if missing else {}
        # Build the result before inserting: a batch can exceed max_prompts.
        result = tuple(rendered[key] if key in rendered else self._prompts[key] for key in keys)
        for key, prompt in zip(keys, result):
            self._put(self._prompts, key, prompt, self._max_prompts)
            self._put(self._lengths, key, len(prompt.input_ids), self._max_lengths)
        return result

    def token_lengths(self, conversations):
        keys = tuple(self._key(messages) for messages in conversations)
        missing = {key: messages for key, messages in zip(keys, conversations)
                   if key not in self._lengths}
        lengths = {key: len(prompt.input_ids) for key, prompt in zip(missing,
            self._template.render_many(tuple(missing.values())), strict=True)} if missing else {}
        result = tuple(lengths[key] if key in lengths else self._lengths[key] for key in keys)
        for key, value in zip(keys, result):
            self._put(self._lengths, key, value, self._max_lengths)
        return result
