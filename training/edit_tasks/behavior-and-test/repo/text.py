"""Text helpers."""
import re


def slugify(title):
    """Lowercase words joined by dashes; anything but letters is dropped."""
    words = re.findall(r"[a-zA-Z]+", title)
    return "-".join(w.lower() for w in words)


def shout(text):
    """Text in capitals with an exclamation mark."""
    return text.upper() + "!"
