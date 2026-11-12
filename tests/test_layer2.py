"""Tests for layer2.py that need NO API key: a fake classifier stands in for Gemini.

Run:  pytest tests/test_layer2.py
"""
from prompt_injection_guard.layer2 import AttackType, Layer2Result, classify, format_examples


class FakeClassifier:
    """Mimics classifier.invoke(); records the inputs it was given."""

    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result, error, None

    def invoke(self, inputs):
        self.seen = inputs
        if self.error:
            raise self.error
        return self.result


def test_format_examples():
    out = format_examples([("attack one", 0.93), ("attack two", 0.9)])
    assert out.splitlines()[0] == '1. "attack one"  (similarity 0.93)'
    assert out.splitlines()[1].startswith('2. "attack two"')


def test_classify_returns_plain_dict():
    fake = FakeClassifier(Layer2Result(is_injection=False, confidence=0.9, attack_type=AttackType.none))
    r = classify(fake, "hello")
    assert r == {"is_injection": False, "confidence": 0.9, "attack_type": "none"}


def test_examples_only_sent_when_given():
    fake = FakeClassifier(Layer2Result(is_injection=True, confidence=1, attack_type=AttackType.other))
    classify(fake, "x")
    assert "examples" not in fake.seen
    classify(fake, "x", [("bad", 0.9)])
    assert "examples" in fake.seen


def test_fails_closed_on_error():
    r = classify(FakeClassifier(error=RuntimeError("quota")), "hello")
    assert r["is_injection"] is True and "quota" in r["error"]


def test_fails_closed_on_empty_response():
    assert classify(FakeClassifier(result=None), "hello")["is_injection"] is True
