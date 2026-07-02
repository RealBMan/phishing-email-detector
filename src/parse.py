import email
from email.message import Message

def extract_body_from_string(raw_email: str) -> str:
    """
    Extracts the body from a raw email string.

    Args:
        raw_email (str): The raw email string.
    """
    
    msg = email.message_from_string(raw_email)
    return extract_body_from_message(msg)

def extract_body_from_message(msg: Message) -> str:
    """
    Extracts the body from an email.message.Message object.

    Args:
        msg (email.message.Message): The email message object.
    """
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload is None:
                    return ""
                try:
                    return payload.decode(charset, errors="replace")
                except LookupError:
                    return payload.decode("utf-8", errors="replace")
        return ""
    charset = msg.get_content_charset() or "utf-8"
    payload = msg.get_payload(decode=True)
    if payload is None:
        return ""
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")