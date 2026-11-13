"""Tests for guard.py. No API key or index needed: Layer 1 and Layer 2 are replaced by fakes.

Run:  pytest tests/test_guard.py
"""
from langchain_core.runnables import RunnableLambda

from prompt_injection_guard import guard
from prompt_injection_guard.guard import REFUSAL_MESSAGE, build_guarded_chain, check_input, guarded_answer


def fake_layer1(verdict):
    """Make a stand-in for layer1.check() that always returns the given verdict."""
    return lambda store, text: {"score": 0.9, "verdict": verdict, "similar_attacks": [("old attack", 0.9)]}


def fake_layer2(is_injection):
    """Make a stand-in for layer2.classify() that always answers is_injection, and records its calls."""
    calls = []

    def fake(classifier, text, similar_attacks=None):
        """The fake classify(): remember the text it was asked about, return the canned answer."""
        calls.append(text)
        return {"is_injection": is_injection, "confidence": 1.0, "attack_type": "other"}

    fake.calls = calls
    return fake


def test_layer1_safe_skips_layer2(monkeypatch):
    """A "safe" Layer 1 verdict passes the message without calling Layer 2."""
    l2 = fake_layer2(True)
    monkeypatch.setattr(guard, "check", fake_layer1("safe"))
    monkeypatch.setattr(guard, "classify", l2)
    r = check_input("hi", None, None)
    assert r["safe"] is True and r["details"]["decided_by"] == "layer1" and l2.calls == []


def test_layer1_injection_skips_layer2(monkeypatch):
    """An "injection" Layer 1 verdict blocks the message without calling Layer 2."""
    l2 = fake_layer2(False)
    monkeypatch.setattr(guard, "check", fake_layer1("injection"))
    monkeypatch.setattr(guard, "classify", l2)
    r = check_input("bad", None, None)
    assert r["safe"] is False and r["details"]["decided_by"] == "layer1" and l2.calls == []


def test_unsure_goes_to_layer2(monkeypatch):
    """An "unsure" verdict is decided by Layer 2, in both directions."""
    monkeypatch.setattr(guard, "check", fake_layer1("unsure"))
    monkeypatch.setattr(guard, "classify", fake_layer2(True))
    r = check_input("hmm", None, None)
    assert r["safe"] is False and r["details"]["decided_by"] == "layer2"
    monkeypatch.setattr(guard, "classify", fake_layer2(False))
    assert check_input("hmm", None, None)["safe"] is True


def test_chain_answers_safe_and_refuses_unsafe(monkeypatch):
    """The full chain returns the main chain's answer for safe input and the refusal for unsafe input."""
    monkeypatch.setattr(guard, "classify", fake_layer2(False))
    main = RunnableLambda(lambda x: "ANSWER: " + x["input"])
    chain = build_guarded_chain(vectorstore=None, classifier=None, main_chain=main)

    monkeypatch.setattr(guard, "check", fake_layer1("safe"))
    assert chain.invoke("hello") == "ANSWER: hello"

    monkeypatch.setattr(guard, "check", fake_layer1("injection"))
    assert chain.invoke("ignore everything") == REFUSAL_MESSAGE


def test_guarded_answer_matches_the_chain(monkeypatch):
    """The plain function gives the same replies as the LangChain chain, plus the verdict."""
    monkeypatch.setattr(guard, "classify", fake_layer2(False))
    main = RunnableLambda(lambda x: "ANSWER: " + x["input"])

    monkeypatch.setattr(guard, "check", fake_layer1("safe"))
    reply, verdict = guarded_answer("hello", None, None, main)
    assert reply == "ANSWER: hello" and verdict["safe"] is True

    monkeypatch.setattr(guard, "check", fake_layer1("injection"))
    reply, verdict = guarded_answer("ignore everything", None, None, main)
    assert reply == REFUSAL_MESSAGE and verdict["safe"] is False
