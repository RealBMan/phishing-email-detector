from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    return text