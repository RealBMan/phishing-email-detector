import os
import sys

# Make ``src`` importable without needing an installed package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import check_ip_url  # noqa: E402
from features import num_links  # noqa: E402
from features import suspicious_tld_links  # noqa: E402

def test_check_ip_url():
    # Test with a URL containing an IP address
    text_with_ip = "Check this link: http://192.168.1.1"
    assert check_ip_url(text_with_ip) == 1

    # Test with a URL containing a domain name
    text_with_domain = "Check this link: http://example.com"
    assert check_ip_url(text_with_domain) == 0

    # Test with multiple IP URLs
    text_with_multiple_ips = "Visit http://192.168.1.1 and https://10.0.0.1"
    assert check_ip_url(text_with_multiple_ips) == 2

def test_check_num_links():
    # Test with a single link
    text_with_link = "Check this link: http://example.com"
    assert num_links(text_with_link) == 1

    # Test with multiple links
    text_with_multiple_links = "Visit http://example.com and https://another.com"
    assert num_links(text_with_multiple_links) == 2

    # Test with no links
    text_without_links = "No links here!"
    assert num_links(text_without_links) == 0

def test_suspicious_tld_links():
    # Single suspicious TLD
    assert suspicious_tld_links("http://secure-login.xyz/verify") == 1

    # Normal TLDs -> none suspicious
    assert suspicious_tld_links("http://paypal.com and http://sourceforge.net") == 0

    # Multiple suspicious TLDs
    assert suspicious_tld_links("visit http://a.xyz and http://b.tk") == 2