"""Step 4: The guard and the guarded chain.

In one line: runs both layers on a message, then either answers it (safe) or refuses it (unsafe).

    user text -> guard (Layer 1, then Layer 2 only if unsure) -> safe?  yes -> main_chain
                                                                        no  -> refusal

The pieces:
    check_input()          one plain function that runs both layers and returns a verdict
    guarded_answer()       the whole flow as plain if/else: check, then answer or refuse
    build_guard()          wraps check_input() so it can be used inside a LangChain chain
    build_main_chain()     a simple Gemini assistant (a placeholder for any real chain, e.g. a RAG chain)
    refusal                returns a fixed "can't help with that" message
    build_guarded_chain()  guard | RunnableBranch(safe -> main_chain, unsafe -> refusal)

Try it live (needs GOOGLE_API_KEY in .env):   python scripts/try_guard.py
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

from prompt_injection_guard.config import LLM_MODEL
from prompt_injection_guard.layer1 import check, load_index
from prompt_injection_guard.layer2 import build_classifier, classify

REFUSAL_MESSAGE = "Sorry, I can't help with that request."


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
def check_input(text, vectorstore, classifier):
    """Run Layer 1, and Layer 2 only when Layer 1 is unsure.

    Returns {"input": text, "safe": True/False, "details": {...}}.
    details says who decided ("layer1" or "layer2") and keeps both layers' outputs for debugging.
    """
    # Step 1: Layer 1 - embed the text and compare it with the known attacks (local, free, fast).
    layer1 = check(vectorstore, text)
    details = {"layer1": layer1, "layer2": None}

    # Step 2: act on Layer 1's verdict.
    if layer1["verdict"] == "injection":      # almost a copy of a known attack: block, no LLM call
        safe, details["decided_by"] = False, "layer1"
    elif layer1["verdict"] == "safe":         # nothing like any known attack: pass, no LLM call
        safe, details["decided_by"] = True, "layer1"
    else:
        # Step 3: unsure - Layer 2 (Gemini) decides, using the similar attacks as evidence.
        layer2 = classify(classifier, text, layer1["similar_attacks"])
        details["layer2"] = layer2
        safe, details["decided_by"] = (not layer2["is_injection"]), "layer2"

    return {"input": text, "safe": safe, "details": details}


def build_guard(vectorstore=None, classifier=None):
    """Turn check_input() into a LangChain Runnable: text in, {"input", "safe", "details"} out."""
    vectorstore = vectorstore or load_index()
    classifier = classifier or build_classifier(use_retrieval=True)
    return RunnableLambda(lambda text: check_input(text, vectorstore, classifier))


# ---------------------------------------------------------------------------
# What happens after the guard
# ---------------------------------------------------------------------------
def build_main_chain():
    """A simple assistant. It receives the guard's output dict and answers the original input."""
    # Imported here so the guard can be tested without the Google package being configured.
    from langchain_google_genai import ChatGoogleGenerativeAI

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer briefly."),
        ("human", "{input}"),
    ])
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return prompt | llm | StrOutputParser()  # the dict's "input" key fills {input}


refusal = RunnableLambda(lambda x: REFUSAL_MESSAGE)


def guarded_answer(text, vectorstore, classifier, main_chain):
    """The whole flow as plain Python: check the message, then answer it or refuse it.

    This does exactly what build_guarded_chain() does, but written as an ordinary if/else
    so it is easy to read and step through. Returns (reply, verdict).
    """
    verdict = check_input(text, vectorstore, classifier)  # Layer 1, then Layer 2 if unsure
    if verdict["safe"]:
        return main_chain.invoke(verdict), verdict        # safe: the assistant answers
    return REFUSAL_MESSAGE, verdict                       # unsafe: fixed refusal, no LLM call


def build_guarded_chain(vectorstore=None, classifier=None, main_chain=None):
    """The same flow as guarded_answer(), built as a LangChain chain:
    guard | RunnableBranch(safe -> main_chain, otherwise -> refusal)"""
    guard = build_guard(vectorstore, classifier)
    main_chain = main_chain or build_main_chain()
    return guard | RunnableBranch(
        (lambda x: x["safe"], main_chain),  # condition, then the chain to run when it is true
        refusal,                            # the last item is the default (unsafe)
    )
