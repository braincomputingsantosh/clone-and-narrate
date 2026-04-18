"""Optional pronunciation respelling dictionary for extract_chapters.py.

English-trained TTS models apply English phonetic rules to every word, which
mangles names and domain terms from other languages. The fix: feed the model
a respelling that uses English letter patterns the model already knows, so
the phonetic guess lands in the right place.

How to use:
    python extract_chapters.py --pdf book.pdf --pronunciations pronunciations.py

The file must define a top-level `PRONUNCIATIONS` dict mapping original text
to respelling. Longer keys are applied first, so compound terms win over
their substrings ("San Francisco" respelled before "San" alone).

Respelling heuristics (for a model trained on English):
    "aa"  -> long /aː/  as in "father"
    "oo"  -> long /uː/  as in "boot"
    "ee"  -> long /iː/  as in "see"
    "ai"  -> /eɪ/       as in "day"
    "u"   -> short /ʌ/  as in "cup"

Examples below are minimal — add entries for names and terms from your
specific domain. Keep the file small (only words the model gets wrong).
"""

from __future__ import annotations

PRONUNCIATIONS: dict[str, str] = {
    # Common English abbreviations that TTS sometimes stumbles on
    "etc.": "et cetera",
    "i.e.": "that is",
    "e.g.": "for example",

    # Example: non-English names (fill in for your domain)
    # "Siddhartha": "Siddaartha",
    # "Château": "Shatoh",
    # "Møller": "Mulluh",
}
