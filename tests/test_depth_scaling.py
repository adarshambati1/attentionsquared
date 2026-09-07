from src.evaluation.gsm8k import extract_answer


def test_extract_gsm8k_answer():
    assert extract_answer("work\n#### 1,234") == "1234"


def test_extract_uses_last_answer_marker():
    assert extract_answer("#### 2\nrevised\n#### 3") == "3"


def test_missing_answer_is_none():
    assert extract_answer("no final answer") is None


def test_comma_group_is_not_split():
    assert extract_answer("Therefore, the answer is $70,000.") == "70000"


def test_truncated_output_does_not_use_intermediate_number():
    text = "The answer is still being calculated after 70,000"
    assert extract_answer(text, allow_fallback=False) is None
