from data.decontaminate import build_eval_index, find_contamination

SHORT_Q = "What position does the player who played for butler cc (ks) play?"          # 12 tokens -> whole-question match
LONG_Q = "How many schools did the player with jersey number three attend before joining the national team?"  # >= 13 tokens
TINY_Q = "How many players?"                                                           # < 8 tokens -> skipped


def index():
    return build_eval_index([SHORT_Q, LONG_Q, TINY_Q])


def test_tiny_questions_are_skipped():
    idx, lookup, skipped = index()
    assert skipped == 1
    assert find_contamination("Reports ask how many players are on the roster.", idx, lookup) is None


def test_whole_short_question_inside_a_document_is_found_ignoring_case_and_punctuation():
    idx, lookup, _ = index()
    doc = "Quiz night results. WHAT position does the player who played for Butler CC (KS) play... answered by Sam."
    assert find_contamination(doc, idx, lookup) == SHORT_Q


def test_a_13_gram_of_a_long_question_is_found():
    idx, lookup, _ = index()
    doc = "intro text " + " ".join(LONG_Q.lower().split()[2:15]) + " trailing text"      # 13 consecutive words
    assert find_contamination(doc, idx, lookup) == LONG_Q


def test_partial_overlap_and_unrelated_text_are_not_flagged():
    idx, lookup, _ = index()
    assert find_contamination("What position does the player play in football?", idx, lookup) is None
    assert find_contamination("The water cycle describes how water moves around the earth " * 5, idx, lookup) is None
    assert find_contamination("", idx, lookup) is None
