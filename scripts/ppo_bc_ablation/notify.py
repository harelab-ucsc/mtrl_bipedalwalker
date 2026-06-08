"""
scripts.ppo_bc_ablation.notify
===============================

Tiny, dependency-free notifier for the end of a sweep. Everything is driven by
the ``notify:`` block of ``sweep.yaml`` (no environment variables):

    notify:
      channel: discord            # discord | ntfy | email | none
      discord:
        webhook_url: "https://discord.com/api/webhooks/..."
      ntfy:
        server: "https://ntfy.sh"
        topic: "my-unique-topic"
      email:
        host: smtp.gmail.com
        port: 587
        user: "you@gmail.com"
        password: "app-password"  # Gmail App Password (needs 2FA)
        to: "lemon@lemonfoxmere.com"

Channel setup, once:
  - discord : target channel -> Edit Channel -> Integrations -> Webhooks ->
              New Webhook -> Copy URL  (no bot, no token — just the URL)
  - ntfy    : pick any topic string, subscribe in the ntfy app or open
              https://ntfy.sh/<topic>  (no account)
  - email   : enable 2FA on the account and create an App Password

Sending is best-effort: any failure (or ``channel: none``/missing config) falls
back to a local macOS ``osascript`` banner and never raises.
"""

from __future__ import annotations

import json
import smtplib
import subprocess
import urllib.request
from email.mime.text import MIMEText


def _local_banner(title: str, message: str) -> None:
    """macOS-only desktop notification; silently ignored elsewhere."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            check=False,
        )
    except FileNotFoundError:
        pass


# Discord (Cloudflare) 403s the default "Python-urllib/x.y" User-Agent, so every
# request must carry a real one — without this, all webhook sends silently fail.
_USER_AGENT = "mtrl-bipedalwalker-notifier/1.0 (+https://github.com)"


def _post(url: str, data: bytes, headers: dict[str, str]) -> None:
    headers = {"User-Agent": _USER_AGENT, **headers}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    urllib.request.urlopen(req, timeout=15).close()


def _send_discord(cfg: dict, title: str, message: str, mention: bool = False) -> None:
    dc = cfg["discord"]
    url = dc["webhook_url"]
    if not url or "webhook" not in url:
        raise ValueError("discord.webhook_url is not set")
    uid = dc.get("mention_user_id")
    prefix = f"<@{uid}> " if (mention and uid) else ""
    payload = {"content": f"{prefix}**{title}**\n{message}"}
    if mention and uid:
        # only ping that one user; never @everyone/@here or roles
        payload["allowed_mentions"] = {"parse": [], "users": [str(uid)]}
    _post(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})


def _send_ntfy(cfg: dict, title: str, message: str, mention: bool = False) -> None:
    nc = cfg["ntfy"]
    server = (nc.get("server") or "https://ntfy.sh").rstrip("/")
    topic = nc["topic"]
    if not topic:
        raise ValueError("ntfy.topic is not set")
    _post(f"{server}/{topic}", message.encode("utf-8"),
          {"Title": title, "Priority": "default"})


def _send_email(cfg: dict, title: str, message: str, mention: bool = False) -> None:
    ec = cfg["email"]
    msg = MIMEText(message)
    msg["Subject"] = title
    msg["From"] = ec["user"]
    msg["To"] = ec["to"]
    with smtplib.SMTP(ec["host"], int(ec.get("port", 587)), timeout=30) as s:
        s.starttls()
        s.login(ec["user"], ec["password"])
        s.send_message(msg)


_SENDERS = {"discord": _send_discord, "ntfy": _send_ntfy, "email": _send_email}


def notify(title: str, message: str, cfg: dict | None, mention: bool = False) -> bool:
    """Send ``title``/``message`` over the configured channel.

    mention: when True, ping the configured user (discord.mention_user_id) — used
    only for important events (routine completions, failures, final summary), not
    intermediate stage updates or the check-in. Ignored by channels without a
    mention concept.

    Returns True if a remote channel accepted it. On any error or ``channel:
    none``/missing config, falls back to a local banner and returns False.
    """
    channel = (cfg or {}).get("channel", "none")
    sender = _SENDERS.get(channel)
    if sender is not None:
        try:
            sender(cfg, title, message, mention=mention)
            return True
        except Exception as e:  # noqa: BLE001 — notification must never crash the run
            print(f"[notify] {channel} send failed ({e}); falling back to local banner")
    _local_banner(title, message)
    return False
