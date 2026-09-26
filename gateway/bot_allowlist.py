"""Which bot and webhook authors a platform admits, by ID.

``{PLATFORM}_ALLOW_BOTS`` (none/mentions/all) decides whether bot-authored messages are read at all,
and an admitted bot skips the human allowlist. That is all-or-nothing: with ``mentions`` on, any bot
or webhook in the server that mentions the agent can instruct it. This narrows it to named IDs:

- ``allowed_bots`` / ``DISCORD_ALLOWED_BOTS``: bot user IDs or webhook IDs (a webhook's messages carry
  the webhook ID as their author ID). When non-empty, only these bot authors are admitted.
- ``bot_allowlist_required`` / ``DISCORD_BOT_ALLOWLIST_REQUIRED``: when true, an EMPTY list admits no
  bot at all. It keeps a deployment fail-closed while the list is still being provisioned (the ID of
  a webhook exists only after it is created).

Human authors are never affected.
"""

from __future__ import annotations

from typing import Any, Optional

from gateway.platforms._shared import decode_json_list_literal as _decode_json_list_literal
from gateway.platforms._shared import extra_or_secret

_TRUE = frozenset({"true", "1", "yes", "on"})

ALLOWED_BOTS_ENV = {"discord": "DISCORD_ALLOWED_BOTS"}
BOT_ALLOWLIST_REQUIRED_ENV = {"discord": "DISCORD_BOT_ALLOWLIST_REQUIRED"}


def parse_id_set(raw: Any) -> set[str]:
    """A YAML list, JSON list literal, or comma-separated string, as a set of stripped IDs."""
    if raw is None:
        return set()
    raw = _decode_json_list_literal(raw)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(part).strip() for part in raw if str(part).strip()}
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def is_truthy(raw: Any) -> bool:
    return str(raw if raw is not None else "").strip().lower() in _TRUE


def bot_author_admitted(allowed_raw: Any, required_raw: Any, author_id: Optional[str]) -> bool:
    """Whether a bot/webhook author passes the ID allowlist.

    No list and not required: every bot passes (the upstream behaviour, unchanged). A list: only its
    IDs pass. Required with no list: nobody passes."""
    allowed = parse_id_set(allowed_raw)
    if not allowed:
        return not is_truthy(required_raw)
    return bool(author_id) and str(author_id).strip() in allowed


def bot_author_admitted_for(platform_value: str, extra: Optional[dict], author_id: Optional[str]) -> bool:
    """``bot_author_admitted`` with the platform's env-over-YAML settings resolved. Platforms with no
    allowlist setting admit every bot, as before."""
    allowed_env = ALLOWED_BOTS_ENV.get(platform_value)
    if not allowed_env:
        return True
    allowed = extra_or_secret(extra, "allowed_bots", allowed_env, "")
    required = extra_or_secret(extra, "bot_allowlist_required", BOT_ALLOWLIST_REQUIRED_ENV[platform_value], "")
    return bot_author_admitted(allowed, required, author_id)
