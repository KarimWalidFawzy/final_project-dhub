"""Automated outputs produced after a research run."""

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Optional


def write_dashboard(result: dict, destination: Path) -> Path:
    """Write machine-readable run metrics for a dashboard to consume."""
    payload = {
        "task": result.get("task"),
        "source_count": len(result.get("sources", [])),
        "agent_count": len(result.get("history", [])),
        "findings": result.get("findings", []),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def send_email(report: str, recipient: str, subject: str = "D-Hub research report") -> None:
    """Send a report through SMTP using environment configuration."""
    host = os.getenv("SMTP_HOST")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM", username or "dhub@localhost")
    if not host:
        raise RuntimeError("SMTP_HOST is required to send email")
    message = EmailMessage()
    message["From"], message["To"], message["Subject"] = sender, recipient, subject
    message.set_content(report)
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=20) as server:
        server.starttls()
        if username and password:
            server.login(username, password)
        server.send_message(message)
