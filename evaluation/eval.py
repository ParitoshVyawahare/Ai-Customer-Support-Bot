"""Evaluation script for the AI Customer Support Bot.

Measures three things across the test set:
  1. Keyword match     — does the answer contain expected keywords?
  2. LLM-as-judge      — does an LLM grader consider the answer correct?
  3. Refusal accuracy  — does the bot correctly refuse off-topic questions?

Usage:
    # Make sure backend is running and docs are uploaded first
    python eval.py
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API = os.getenv("BACKEND_URL", "http://localhost:8000")
JUDGE_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

groq_client = Groq(api_key=GROQ_KEY)

REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "no information",
]


def keyword_score(answer: str, expected_keywords: list) -> float:
    """Returns fraction of expected keywords found in the answer (0.0 to 1.0)."""
    if not expected_keywords:
        return 1.0
    a = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in a)
    return hits / len(expected_keywords)


def is_refusal(answer: str) -> bool:
    """Detects whether the bot refused to answer."""
    a = answer.lower()
    return any(phrase in a for phrase in REFUSAL_PHRASES)


def llm_judge(question: str, expected: str, actual: str) -> int:
    """Asks an LLM to grade the answer. Returns 1 if correct, 0 otherwise."""
    prompt = (
        "You are evaluating an AI assistant's answer for correctness.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {expected}\n"
        f"Assistant's answer: {actual}\n\n"
        "Does the assistant's answer contain the key information from the reference "
        "answer? Minor wording differences are fine. The answer is correct if a user "
        "would get the right information.\n\n"
        "Respond with ONLY a single character: 1 for correct, 0 for incorrect."
    )
    try:
        resp = groq_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        out = resp.choices[0].message.content.strip()
        return 1 if out.startswith("1") else 0
    except Exception as e:
        print(f"  judge error: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(Path(__file__).parent / "test_questions.json"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "eval_results.json"))
    parser.add_argument("--session", default="eval-run")
    parser.add_argument("--sleep", type=float, default=0.4, help="Sleep between requests (rate limit)")
    args = parser.parse_args()

    # Sanity check
    try:
        h = requests.get(f"{API}/health", timeout=10).json()
        if h["doc_count"] == 0:
            print("ERROR: No documents in the knowledge base.")
            print("Upload data/sample_docs/techflow_knowledge_base.txt first via the UI or /upload.")
            sys.exit(1)
        print(f"API healthy. {h['doc_count']} chunks in DB. Model: {h['model']}\n")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API}: {e}")
        sys.exit(1)

    with open(args.questions) as f:
        tests = json.load(f)

    # Clear memory for clean eval (each question is independent)
    requests.post(f"{API}/clear", json={"session_id": args.session}, timeout=10)

    results = []
    kw_sum = 0.0
    judge_correct = 0
    answer_n = 0
    refusal_n = 0
    refused_correctly = 0

    print(f"Running {len(tests)} test questions\n")
    print(f"{'#':>3}  {'Type':>8}  {'KW':>5}  {'Judge':>5}  Question")
    print("-" * 100)

    for i, t in enumerate(tests, start=1):
        # Clear memory before each question so they don't influence each other
        requests.post(f"{API}/clear", json={"session_id": args.session}, timeout=10)

        try:
            r = requests.post(
                f"{API}/ask",
                json={"question": t["question"], "session_id": args.session},
                timeout=120,
            )
            answer = r.json()["answer"] if r.ok else f"[ERROR: {r.status_code} {r.text}]"
            confidence = r.json().get("confidence") if r.ok else None
            sources = r.json().get("sources", []) if r.ok else []
        except Exception as e:
            answer = f"[REQUEST FAILED: {e}]"
            confidence = None
            sources = []

        if t.get("expect_refusal"):
            refusal_n += 1
            refused = is_refusal(answer)
            kw = 1.0 if refused else 0.0
            judge = 1 if refused else 0
            if refused:
                refused_correctly += 1
            kind = "refusal"
        else:
            answer_n += 1
            kw = keyword_score(answer, t.get("expected_keywords", []))
            judge = llm_judge(t["question"], t["expected_answer"], answer)
            kw_sum += kw
            judge_correct += judge
            kind = "answer"

        results.append({
            "question": t["question"],
            "expected": t.get("expected_answer", "[refusal expected]"),
            "actual": answer,
            "kind": kind,
            "keyword_score": kw,
            "judge_correct": judge,
            "confidence": confidence,
            "cited_sources": [s["source"] for s in sources if s.get("cited")],
        })

        q_short = t["question"][:60] + ("..." if len(t["question"]) > 60 else "")
        print(f"{i:>3}  {kind:>8}  {kw:>5.2f}  {judge:>5}  {q_short}")
        time.sleep(args.sleep)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    if answer_n:
        print(f"Answer questions: {answer_n}")
        print(f"  Keyword score (avg): {kw_sum / answer_n:.1%}")
        print(f"  LLM-judge accuracy:  {judge_correct / answer_n:.1%}")
    if refusal_n:
        print(f"Refusal questions: {refusal_n}")
        print(f"  Correctly refused:   {refused_correctly}/{refusal_n} ({refused_correctly / refusal_n:.1%})")

    overall = (judge_correct + refused_correctly) / len(tests)
    print(f"\nOverall accuracy: {overall:.1%}  ({judge_correct + refused_correctly}/{len(tests)})")

    summary = {
        "total": len(tests),
        "overall_accuracy": round(overall, 4),
        "answer_keyword_avg": round(kw_sum / answer_n, 4) if answer_n else None,
        "answer_judge_accuracy": round(judge_correct / answer_n, 4) if answer_n else None,
        "refusal_accuracy": round(refused_correctly / refusal_n, 4) if refusal_n else None,
        "model": h["model"],
    }

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()