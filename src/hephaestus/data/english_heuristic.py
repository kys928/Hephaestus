"""Original conservative source-screen language heuristic; not certification."""
import re
import unicodedata

STOPWORDS = frozenset(
    "a an the of for to from is are was were in on at with by and or as that this "
    "it can will which what where when who why how not they their we you your "
    "have has do does please output return answer provide".split()
)


def english_likelihood(text: str) -> tuple[bool, str]:
    """Transparent conservative heuristic, NOT language-ID certification."""
    if not text or len(text.strip()) < 12:
        return False, "too_short"
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", text):
        return False, "cjk_script_detected"
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 7:
        return False, "too_few_letters"
    latin = sum("LATIN" in unicodedata.name(ch, "") for ch in letters)
    if latin / len(letters) < 0.90:
        return False, "non_latin_script"
    words = re.findall(r"[A-Za-z]+", text.casefold())
    if len(words) < 5 or not set(words).intersection(STOPWORDS):
        return False, "weak_english_word_evidence"
    return True, "latin_with_english_function_words"


