"""Tests for layer2.py that need NO API key: a fake classifier stands in for Gemini.

Run:  pytest tests/test_layer2.py
"""
from prompt_injection_guard.layer2 import AttackType, Layer2Result, classify, format_examples


class FakeClassifier:
    """Mimics classifier.invoke(); records the inputs it was given."""

    def __init__(self, result=None, error=None):
        """Set the canned result to return, or an error to raise."""
        self.result, self.error, self.seen = result, error, None

    def invoke(self, inputs):
        """Remember what we were given, then return the canned result or raise the canned error."""
        self.seen = inputs
        if self.error:
            raise self.error
        return self.result


def test_format_examples():
    """The retrieved attacks are turned into a numbered list with their similarity."""
    out = format_examples([("attack one", 0.93), ("attack two", 0.9)])
    assert out.splitlines()[0] == '1. "attack one"  (similarity 0.93)'
    assert out.splitlines()[1].startswith('2. "attack two"')


def test_classify_returns_plain_dict():
    """classify() converts the structured result into a plain dict."""
    fake = FakeClassifier(Layer2Result(is_injection=False, confidence=0.9, attack_type=AttackType.none))
    r = classify(fake, "hello")
    assert r == {"is_injection": False, "confidence": 0.9, "attack_type": "none"}


def test_examples_only_sent_when_given():
    """The examples are sent to the classifier only when similar attacks are passed in."""
    fake = FakeClassifier(Layer2Result(is_injection=True, confidence=1, attack_type=AttackType.other))
    classify(fake, "x")
    assert "examples" not in fake.seen
    classify(fake, "x", [("bad", 0.9)])
    assert "examples" in fake.seen


def test_fails_closed_on_error():
    """If the classifier raises an error, the message is blocked."""
    r = classify(FakeClassifier(error=RuntimeError("quota")), "hello")
    assert r["is_injection"] is True and "quota" in r["error"]


def test_fails_closed_on_empty_response():
    """If the classifier returns nothing, the message is blocked."""
    assert classify(FakeClassifier(result=None), "hello")["is_injection"] is True
