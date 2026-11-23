"""Train the tiny intent-routing MLP that powers alexmueller07.github.io.

Architecture: 128 hashed char-trigram/word features -> 20 relu -> 14 relu -> 7 softmax.
Weights are exported to js/weights.json and executed in the browser with
hand-written matrix math. Feature hashing here must match js/main.js exactly.

Run:  python train/train.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

N_BUCKETS = 128
CLASSES = ["who", "research", "github", "linkedin", "email", "help", "ood"]

DATA: dict[str, list[str]] = {
    "who": [
        "who", "who?", "who are you", "who is this", "who is alexander",
        "who is alexander mueller", "tell me about yourself", "about",
        "about you", "introduce yourself", "what is your name", "your name",
        "name", "hello", "hi", "hey", "whats up", "tell me about alex",
        "whose site is this", "who owns this site", "what do you do", "bio",
        "background", "tell me who you are", "describe yourself", "intro",
        "say hello", "greetings", "yo", "what are you", "are you alexander",
        "alex", "alexander", "alexander mueller", "whats your story",
        "where are you from", "where do you live", "where are you based",
        "tell me about you", "give me your bio", "who am i talking to",
    ],
    "research": [
        "research", "what research do you do", "what do you research",
        "research interests", "what are your research interests",
        "ai research", "machine learning", "tell me about your research",
        "what are you working on", "current work", "what are you studying",
        "areas of interest", "interests", "vision language models",
        "continual learning", "autonomous navigation", "robotics",
        "what topics", "academic interests", "papers", "publications",
        "what fields", "deep learning", "computer vision",
        "what kind of ai do you do", "research areas", "what labs",
        "what do you study", "academics", "what science do you do",
        "research topics", "your studies", "what ml topics",
        "fields of study", "scientific interests",
    ],
    "github": [
        "github", "git", "code", "show me your code", "where is your code",
        "projects", "your projects", "repos", "repositories", "portfolio",
        "what have you built", "show me projects", "source code", "coding",
        "programming", "what languages do you code in", "rust", "python",
        "software", "side projects", "open source", "show code",
        "link to github", "github profile", "what do you build", "dev work",
        "programming projects", "code samples", "show me what you made",
        "your repositories", "coding projects", "github link",
    ],
    "linkedin": [
        "linkedin", "link to linkedin", "linkedin profile", "job", "jobs",
        "hire", "hiring", "can i hire you", "are you looking for work",
        "where do you work", "work experience", "experience", "employment",
        "career", "resume", "cv", "internship", "intern", "recruiter",
        "recruiting", "professional profile", "positions",
        "what jobs have you had", "current job", "looking for internships",
        "open to work", "employment history", "your experience",
        "work history", "professional background", "are you employed",
        "what is your job", "do you have a job", "who do you work for",
    ],
    "email": [
        "email", "e mail", "contact", "contact info", "how do i contact you",
        "how can i reach you", "reach out", "get in touch",
        "send you a message", "message you", "talk to you", "mail",
        "email address", "whats your email", "your email", "connect",
        "chat with you", "communicate", "reach you", "write to you",
        "drop a line", "contact details", "how to reach you",
        "speak with you", "can we talk", "i want to contact you",
        "send an email", "shoot you a message", "get a hold of you",
    ],
    "help": [
        "help", "help?", "how does this work", "how does this site work",
        "what is this", "what is this site", "how was this built",
        "how did you build this", "is this a real neural network",
        "is this real", "explain this site", "explain how this works",
        "what am i looking at", "how does the neural net work",
        "what is this thing", "instructions", "how do i use this", "usage",
        "what can i ask", "what can you do", "commands", "options", "menu",
        "how this was made", "tech stack", "what framework",
        "how do you work", "are you a real ai", "are you an llm",
        "how were you trained", "what model is this", "does this work",
        "does this searchbar work", "does the search work",
        "is this actually a neural network", "what powers this",
        "how does the search bar work", "what architecture is this",
        "how many parameters do you have", "explain yourself",
        "how is this implemented",
    ],
    "ood": [
        "what is the meaning of life", "tell me a joke", "whats the weather",
        "weather today", "what time is it", "sing a song", "write me a poem",
        "who is the president", "capital of france",
        "how tall is mount everest", "best pizza near me",
        "stock market today", "translate this to french",
        "what is two plus two", "play music", "random", "asdf", "qwerty",
        "lorem ipsum", "blah blah", "banana", "do you dream",
        "are you alive", "what is love", "how do magnets work",
        "recommend a movie", "favorite color", "tell me a secret",
        "what should i eat", "sports scores", "news today", "bitcoin price",
        "how to cook pasta", "is water wet", "why is the sky blue",
        "fhqwhgads", "zzzzz", "hmmmm", "ok", "testing 123",
        "what day is it", "how far is the moon", "give me a recipe",
        "solve my homework", "what is quantum physics",
    ],
}

TEST_QUERIES = [
    "does this searchbar work", "where do you work", "who are you",
    "show me your rust code", "how can i email you", "what do you research",
    "how was this website made", "what is the airspeed of a swallow",
    "can i recruit you for an internship", "yo what up",
]


def fnv1a(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def featurize(text: str) -> np.ndarray:
    x = np.zeros(N_BUCKETS)
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()
    if not norm:
        return x
    for tok in norm.split(" "):
        x[fnv1a("w:" + tok) % N_BUCKETS] += 1
        padded = "^" + tok + "$"
        for i in range(len(padded) - 2):
            x[fnv1a("c:" + padded[i : i + 3]) % N_BUCKETS] += 1
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def main() -> None:
    rng = np.random.default_rng(7)

    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for phrase in DATA[cls]:
            X.append(featurize(phrase))
            y.append(ci)
    X = np.array(X)
    y = np.array(y)
    n, k = len(X), len(CLASSES)
    print(f"dataset: {n} phrases, {k} classes, {N_BUCKETS} feature buckets")

    h1, h2 = 20, 14
    W1 = rng.normal(0, np.sqrt(2 / N_BUCKETS), (N_BUCKETS, h1))
    b1 = np.zeros(h1)
    W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2))
    b2 = np.zeros(h2)
    W3 = rng.normal(0, np.sqrt(2 / h2), (h2, k))
    b3 = np.zeros(k)

    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    lr, beta1, beta2, eps, l2 = 0.02, 0.9, 0.999, 1e-8, 1e-4
    Y = np.eye(k)[y]

    for epoch in range(1, 2501):
        a1 = np.maximum(0, X @ W1 + b1)
        a2 = np.maximum(0, a1 @ W2 + b2)
        logits = a2 @ W3 + b3
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)

        if epoch % 500 == 0 or epoch == 1:
            loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
            acc = (probs.argmax(axis=1) == y).mean()
            print(f"epoch {epoch:5d}  loss {loss:.4f}  acc {acc:.3f}")

        d_logits = (probs - Y) / n
        dW3 = a2.T @ d_logits + l2 * W3
        db3 = d_logits.sum(axis=0)
        d_a2 = d_logits @ W3.T * (a2 > 0)
        dW2 = a1.T @ d_a2 + l2 * W2
        db2 = d_a2.sum(axis=0)
        d_a1 = d_a2 @ W2.T * (a1 > 0)
        dW1 = X.T @ d_a1 + l2 * W1
        db1 = d_a1.sum(axis=0)

        for p, g, mi, vi in zip(params, [dW1, db1, dW2, db2, dW3, db3], m, v):
            mi[:] = beta1 * mi + (1 - beta1) * g
            vi[:] = beta2 * vi + (1 - beta2) * g * g
            mh = mi / (1 - beta1**epoch)
            vh = vi / (1 - beta2**epoch)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def predict(q: str) -> tuple[str, float]:
        a1 = np.maximum(0, featurize(q) @ W1 + b1)
        a2 = np.maximum(0, a1 @ W2 + b2)
        logits = a2 @ W3 + b3
        ex = np.exp(logits - logits.max())
        p = ex / ex.sum()
        return CLASSES[int(p.argmax())], float(p.max())

    print("\nspot checks:")
    for q in TEST_QUERIES:
        cls, p = predict(q)
        print(f"  {q!r:50s} -> {cls:9s} p={p:.2f}")

    n_params = sum(p.size for p in params)
    out = {
        "classes": CLASSES,
        "buckets": N_BUCKETS,
        "params": int(n_params),
        "W1": np.round(W1, 4).tolist(),
        "b1": np.round(b1, 4).tolist(),
        "W2": np.round(W2, 4).tolist(),
        "b2": np.round(b2, 4).tolist(),
        "W3": np.round(W3, 4).tolist(),
        "b3": np.round(b3, 4).tolist(),
    }
    dest = Path(__file__).resolve().parent.parent / "js" / "weights.json"
    dest.write_text(json.dumps(out, separators=(",", ":")))
    print(f"\nwrote {dest} ({n_params} parameters, {dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
