from hephaestus.data.normalization import normalize_record


def test_input_output_pair_normalizes_to_prompt_target() -> None:
    assert normalize_record({"input": "  Follow this  ", "output": "  Answer here  "}) == {
        "prompt": "Follow this",
        "target": "Answer here",
    }


def test_instruction_response_pair_normalizes_to_prompt_target() -> None:
    assert normalize_record({"instruction": "Do this", "response": "Done"}) == {
        "prompt": "Do this",
        "target": "Done",
    }


def test_question_answer_pair_normalizes_to_prompt_target() -> None:
    assert normalize_record({"question": "Why?", "answer": "Because."}) == {
        "prompt": "Why?",
        "target": "Because.",
    }


def test_partial_instruction_pair_is_rejected() -> None:
    assert normalize_record({"input": "Prompt only", "output": ""}) is None


def test_existing_prompt_target_and_text_behavior_remains() -> None:
    assert normalize_record({"prompt": "P", "target": "T"}) == {"prompt": "P", "target": "T"}
    assert normalize_record({"text": "plain text"}) == {"text": "plain text"}
