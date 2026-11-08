"""Tool hooks. Email and web lookup work out of the box; site actions are the seam you extend."""
import email
import os
import re
import imaplib
import smtplib
from email.message import EmailMessage

import httpx


def send_email(to: str, subject: str, body: str) -> str:
    user = os.environ["EMAIL_ADDRESS"]
    pw = os.environ["EMAIL_APP_PASSWORD"]
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    msg = EmailMessage()
    msg["To"], msg["From"], msg["Subject"] = to, user, subject
    msg.set_content(body)
    with smtplib.SMTP(host, 587) as s:
        s.starttls()
        s.login(user, pw)
        s.send_message(msg)
    return f"sent to {to}"


def read_recent_emails(count: int = 5) -> str:
    user = os.environ["EMAIL_ADDRESS"]
    pw = os.environ["EMAIL_APP_PASSWORD"]
    host = os.environ.get("IMAP_HOST", "imap.gmail.com")
    mail = imaplib.IMAP4_SSL(host)
    mail.login(user, pw)
    mail.select("INBOX")
    _, data = mail.search(None, "ALL")
    ids = data[0].split()[-int(count):][::-1]
    out = []
    for i in ids:
        _, raw = mail.fetch(i, "(RFC822)")
        m = email.message_from_bytes(raw[0][1])
        text = re.sub(r"\s+", " ", m.get_payload() if isinstance(m.get_payload(), str)
                      else "(mixed parts)")
        out.append(f"From {m['From']}: {m['Subject']}: {text[:600]}")
    mail.logout()
    return " || ".join(out) or "no emails"


def web_lookup(url: str) -> str:
    r = httpx.get(url, timeout=30, follow_redirects=True,
                  headers={"User-Agent": "Mozilla/5.0"})
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", r.text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def web_action(site: str, action: str, details: str) -> str:
    """Seam for site flows and account logins. Wire Playwright here: launch a browser,
    sign in with that user's stored credentials, and drive the site. Nothing here is live yet."""
    raise NotImplementedError(
        f"web_action not wired yet: site={site} action={action} details={details}")


ROUTES = {
    "send_email": send_email,
    "read_recent_emails": read_recent_emails,
    "web_lookup": web_lookup,
    "web_action": web_action,
}
