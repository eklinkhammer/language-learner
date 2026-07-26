"""Mapping of IPA phonemes to English example words.

Used to give learners a familiar auditory anchor for unfamiliar sounds,
e.g. '/e/ — like the "e" in "bed"'.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhonemeExample:
    word: str        # English example word containing the phoneme
    highlight: str   # The letter(s) in the word that produce the sound
    description: str # e.g. 'like the "e" in "bed"'


PHONEME_EXAMPLES: dict[str, PhonemeExample | None] = {
    # --- Vowels ---
    "i":  PhonemeExample("see",   "ee",  'like the "ee" in "see"'),
    "ɪ":  PhonemeExample("bit",   "i",   'like the "i" in "bit"'),
    "e":  PhonemeExample("bed",   "e",   'like the "e" in "bed"'),
    "ɛ":  PhonemeExample("bet",   "e",   'like the "e" in "bet"'),
    "æ":  PhonemeExample("cat",   "a",   'like the "a" in "cat"'),
    "ɑ":  PhonemeExample("father","a",   'like the "a" in "father"'),
    "ɒ":  PhonemeExample("lot",   "o",   'like the "o" in "lot" (British)'),
    "ɔ":  PhonemeExample("law",   "aw",  'like the "aw" in "law"'),
    "o":  PhonemeExample("go",    "o",   'like the "o" in "go"'),
    "ʊ":  PhonemeExample("book",  "oo",  'like the "oo" in "book"'),
    "u":  PhonemeExample("food",  "oo",  'like the "oo" in "food"'),
    "ʌ":  PhonemeExample("cup",   "u",   'like the "u" in "cup"'),
    "ə":  PhonemeExample("about", "a",   'like the "a" in "about"'),
    "a":  PhonemeExample("father","a",   'like the "a" in "father"'),
    # --- Consonants ---
    "p":  PhonemeExample("pin",   "p",   'like the "p" in "pin"'),
    "b":  PhonemeExample("bat",   "b",   'like the "b" in "bat"'),
    "t":  PhonemeExample("top",   "t",   'like the "t" in "top"'),
    "d":  PhonemeExample("dog",   "d",   'like the "d" in "dog"'),
    "k":  PhonemeExample("cat",   "c",   'like the "c" in "cat"'),
    "ɡ":  PhonemeExample("go",    "g",   'like the "g" in "go"'),
    "f":  PhonemeExample("fun",   "f",   'like the "f" in "fun"'),
    "v":  PhonemeExample("van",   "v",   'like the "v" in "van"'),
    "θ":  PhonemeExample("think", "th",  'like the "th" in "think"'),
    "ð":  PhonemeExample("the",   "th",  'like the "th" in "the"'),
    "s":  PhonemeExample("sun",   "s",   'like the "s" in "sun"'),
    "z":  PhonemeExample("zoo",   "z",   'like the "z" in "zoo"'),
    "ʃ":  PhonemeExample("she",   "sh",  'like the "sh" in "she"'),
    "ʒ":  PhonemeExample("vision","si",  'like the "si" in "vision"'),
    "h":  PhonemeExample("hat",   "h",   'like the "h" in "hat"'),
    "m":  PhonemeExample("man",   "m",   'like the "m" in "man"'),
    "n":  PhonemeExample("net",   "n",   'like the "n" in "net"'),
    "ŋ":  PhonemeExample("sing",  "ng",  'like the "ng" in "sing"'),
    "l":  PhonemeExample("let",   "l",   'like the "l" in "let"'),
    "ɹ":  PhonemeExample("red",   "r",   'like the "r" in "red"'),
    "r":  PhonemeExample("red",   "r",   'like the "r" in "red"'),
    "j":  PhonemeExample("yes",   "y",   'like the "y" in "yes"'),
    "w":  PhonemeExample("wet",   "w",   'like the "w" in "wet"'),
    "tʃ": PhonemeExample("chip",  "ch",  'like the "ch" in "chip"'),
    "dʒ": PhonemeExample("judge", "j",   'like the "j" in "judge"'),
    # --- Spanish-specific (no English equivalent) ---
    "ɾ":  None,
    "β":  None,
    "ɣ":  None,
    "ɲ":  None,
    "x":  None,
}


def get_example(phoneme: str) -> PhonemeExample | None:
    """Look up the English example for an IPA phoneme, or None."""
    return PHONEME_EXAMPLES.get(phoneme)
