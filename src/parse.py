import email

def extract_body_from_string(raw_email: str) -> str:
    """
    Extracts the body from a raw email string.

    Args:
        raw_email (str): The raw email string.
    """
    
    msg = email.message_from_string(raw_email)
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/plain" and "attachment" not in content_disposition:
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload is None:
                    return ""
                return payload.decode(charset, errors="replace")
        return ""
    charset = msg.get_content_charset() or "utf-8"
    payload = msg.get_payload(decode=True)
    if payload is None:
        return ""
    return payload.decode(charset, errors="replace")