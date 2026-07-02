import re
from urllib.parse import urlparse

# TLDs disproportionately abused for phishing (cheap/free registration, weak
# oversight). A set for fast membership checks.
SUSPICIOUS_TLDS = {"zip", "xyz", "top", "tk", "ru", "gq", "ml", "cf", "ga",
                   "cn", "work", "click"}

URL_PATTERN = r"https?://[^\s]+"


def check_ip_url(text: str) -> int:
    pattern = r"https?://\d+\.\d+\.\d+\.\d+"
    matches = re.findall(pattern, text)
    return len(matches)


def num_links(text: str) -> int:
    matches = re.findall(URL_PATTERN, text)
    return len(matches)


def suspicious_tld_links(text: str) -> int:
    """Count links whose domain ends in a suspicious TLD — a *qualitative*
    signal (is the link dangerous?) rather than a count of all links."""
    count = 0
    for url in re.findall(URL_PATTERN, text):
        try:
            host = urlparse(url).hostname  # 'secure-login.xyz'; None if empty
        except ValueError:
            continue                       # malformed URL (e.g. bad IPv6) in real data
        if not host:
            continue
        tld = host.rsplit(".", 1)[-1].lower()   # last dot-segment -> 'xyz'
        if tld in SUSPICIOUS_TLDS:
            count += 1
    return count