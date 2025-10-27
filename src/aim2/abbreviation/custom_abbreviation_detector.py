from typing import Tuple, List, Optional, Set, Dict
from collections import defaultdict
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
from spacy.language import Language

STOP_WORDS = {"and", "for", "in", "of", "the", "an"}

def find_abbreviation(
    long_form_candidate: Span, short_form_candidate: Span
) -> Tuple[Span, Optional[Span]]:
    """
    A more robust implementation of an abbreviation detection algorithm.
    This algorithm iterates through all possible start points in the long form
    and scores them based on how well they match the short form.
    """
    short_form_text = "".join(short_form_candidate.text.split())
    long_form_tokens = [token for token in long_form_candidate]

    s_len = len(short_form_text)
    l_len = len(long_form_tokens)

    if s_len == 0 or l_len == 0:
        return short_form_candidate, None

    best_score = -1
    best_match_start_index = -1

    for i in range(l_len):
        short_form_char_index = 0
        current_score = 0

        for j in range(i, l_len):
            long_word = long_form_tokens[j].text
            long_word_lower = long_word.lower()

            if short_form_char_index == s_len:
                break

            if long_word_lower in STOP_WORDS and j > i:
                continue

            long_form_char_index_in_word = 0
            while (
                short_form_char_index < s_len
                and long_form_char_index_in_word < len(long_word)
            ):
                short_char = short_form_text[short_form_char_index].lower()
                long_char = long_word_lower[long_form_char_index_in_word]

                if short_char == long_char:
                    if long_form_char_index_in_word == 0:
                        current_score += 2
                    else:
                        current_score += 1
                    short_form_char_index += 1
                long_form_char_index_in_word += 1

        if (
            short_form_char_index == s_len and
            current_score > best_score and
            current_score >= s_len and
            short_form_text[0].lower() == long_form_tokens[i].text[0].lower()
        ):
            best_score = current_score
            best_match_start_index = i

    if best_match_start_index != -1:
        return short_form_candidate, long_form_candidate[best_match_start_index:]

    return short_form_candidate, None


def filter_matches(
    matcher_output: List[Tuple[int, int, int]], doc: Doc
) -> List[Tuple[Span, Span]]:
    candidates = []
    for match in matcher_output:
        start = match[1]
        end = match[2]
        
        short_form_candidate = doc[start:end]

        max_len = len(short_form_candidate) + 5
        search_start = max(0, start - max_len - 1)
        
        long_form_start_index = start - 1
        for i in range(start - 2, search_start -1, -1):
            token = doc[i]
            if token.is_title or token.is_sent_start or token.is_digit:
                long_form_start_index = i
                break
            if token.is_punct:
                long_form_start_index = i + 1
                break
        
        long_form_candidate = doc[long_form_start_index : start - 1]

        if len(long_form_candidate) > 1:
            first_token = long_form_candidate[0]
            if len(first_token.text) == 1 and first_token.is_upper and not first_token.is_sent_start:
                long_form_candidate = long_form_candidate[1:]

        if short_form_filter(short_form_candidate) and len(long_form_candidate) > 0:
            candidates.append((long_form_candidate, short_form_candidate))

    return candidates


def short_form_filter(span: Span) -> bool:
    """
    Filters short form candidates to reject obvious non-abbreviations.
    """
    text = span.text
    # Reject if it's too long or too short.
    if len(text) > 10 or len(text) < 1:
        return False
    
    # FINAL FIX: If the candidate is very short (1-2 chars), it must contain an uppercase letter.
    # This rejects list markers like (a), (b), (i), (iv), etc.
    if len(text) <= 2 and not any(c.isupper() for c in text):
        return False

    # The first character should be a letter.
    if not text[0].isalpha():
        return False
        
    return True


@Language.factory("custom_abbreviation_detector")
class AbbreviationDetector:
    """
    Detects abbreviations using a robust algorithm.
    This class sets the `._.abbreviations` attribute on spaCy Doc.
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "custom_abbreviation_detector",
    ) -> None:
        Doc.set_extension("abbreviations", default=[], force=True)
        Span.set_extension("long_form", default=None, force=True)

        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("parenthesis", [[{"ORTH": "("}, {"OP": "+"}, {"ORTH": ")"}]])

    def __call__(self, doc: Doc) -> Doc:
        matches = self.matcher(doc)
        matches_no_brackets = [(x[0], x[1] + 1, x[2] - 1) for x in matches]
        
        abbreviations = {}

        filtered_candidates = filter_matches(matches_no_brackets, doc)
        for long_candidate, short_candidate in filtered_candidates:
            short, long = find_abbreviation(long_candidate, short_candidate)
            
            if long is not None and short.text not in abbreviations:
                # Final check to prevent nonsensical short definitions
                if len(long.text) > len(short.text):
                    abbreviations[short.text] = long
                    short._.long_form = long
                    doc._.abbreviations.append(short)
                
        return doc