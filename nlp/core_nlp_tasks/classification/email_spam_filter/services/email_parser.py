"""Safe .eml parsing that extracts text without loading remote resources."""

from email import policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup


MAX_EMAIL_SIZE = 10 * 1024 * 1024


def _read_bytes(source):
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, bytearray):
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > MAX_EMAIL_SIZE:
            raise ValueError("Email file exceeds the 10 MiB safety limit.")
        data = path.read_bytes()
    elif hasattr(source, "read"):
        data = source.read(MAX_EMAIL_SIZE + 1)
    else:
        raise TypeError("Email source must be bytes, a file path, or a readable file object.")
    if len(data) > MAX_EMAIL_SIZE:
        raise ValueError("Email file exceeds the 10 MiB safety limit.")
    return data


def _part_text(part):
    content = part.get_content()
    if isinstance(content, bytes):
        return content.decode(part.get_content_charset() or "utf-8", errors="replace")
    return str(content)


def parse_eml(source):
    """Parse email headers and text/HTML bodies from bytes, a path, or upload."""
    message = BytesParser(policy=policy.default).parsebytes(_read_bytes(source))
    plain_parts = []
    html_parts = []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_part_text(part))
        elif content_type == "text/html":
            html_parts.append(_part_text(part))

    html_body = "\n".join(html_parts)
    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_body:
        body = BeautifulSoup(html_body, "html.parser").get_text(" ", strip=True)
    else:
        body = ""
    return {
        "subject": str(message.get("subject", "")),
        "sender": str(message.get("from", "")),
        "body": body,
        "html_body": html_body,
    }
