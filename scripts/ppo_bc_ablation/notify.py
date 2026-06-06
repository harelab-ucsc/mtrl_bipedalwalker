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


def _post(url: str, data: bytes, headers: dict[str, str]) -> None:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    urllib.request.urlopen(req, timeout=15).close()


def _send_discord(cfg: dict, title: str, message: str) -> None:
    url = cfg["discord"]["webhook_url"]
    if not url or "webhook" not in url:
        raise ValueError("discord.webhook_url is not set")
    body = json.dumps({"content": f"**{title}**\n{message}"}).encode()
    _post(url, body, {"Content-Type": "application/json"})


def _send_ntfy(cfg: dict, title: str, message: str) -> None:
    nc = cfg["ntfy"]
    server = (nc.get("server") or "https://ntfy.sh").rstrip("/")
    topic = nc["topic"]
    if not topic:
        raise ValueError("ntfy.topic is not set")
    _post(f"{server}/{topic}", message.encode("utf-8"),
          {"Title": title, "Priority": "default"})


def _send_email(cfg: dict, title: str, message: str) -> None:
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


def notify(title: str, message: str, cfg: dict | None) -> bool:
    """Send ``title``/``message`` over the configured channel.

    Returns True if a remote channel accepted it. On any error or ``channel:
    none``/missing config, falls back to a local banner and returns False.
    """
    channel = (cfg or {}).get("channel", "none")
    sender = _SENDERS.get(channel)
    if sender is not None:
        try:
            sender(cfg, title, message)
            return True
        except Exception as e:  # noqa: BLE001 — notification must never crash the run
            print(f"[notify] {channel} send failed ({e}); falling back to local banner")
    _local_banner(title, message)
    return False
