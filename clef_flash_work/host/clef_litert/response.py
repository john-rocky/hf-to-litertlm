"""Head logits -> the SystemOne-compatible response body, as the checkpoint's own `systemone()` builds it.

Probabilities: per question, the float32 softmax of its option logits, in NumPy's order (exp of z - max z, a float32
sum, a float32 division); the author's `question_logits.float().softmax(-1)` is the same computation in torch. They
enter Python as floats (`.tolist()`), as in the author's code, and `systemone_answer` is the author's function:

    noul    {"type": "noul", "noul": round(p(true), 4)}
    choice  {"type": "choice", "choice": the most probable option in the request's criteria order (first on a tie),
             "confidence": round(p(choice), 4), "probabilities": {option: round(p, 4)} in the criteria order}
    score   {"type": "score", "score": round(sum(level * p(level)), 4) (Python's left-to-right sum),
             "confidence": round(max p, 4), "legend": {level: criterion}, "probabilities": {level: round(p, 4)}}

Body: {"model": request["model"], "answers": {question id: answer} in the request's order,
       "usage": {"input_tokens": the record's tokens, "output_tokens": 0}}."""
from __future__ import annotations

import numpy as np


def question_probs(logits) -> np.ndarray:
    """float32 softmax over one question's option logits."""
    z = np.asarray(logits, dtype=np.float32)
    e = np.exp(z - z.max())
    return e / e.sum(dtype=np.float32)


def systemone_answer(question: dict, probabilities: dict) -> dict:
    """Per-option probabilities of one question -> its SystemOne answer (the author's function)."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": round(probabilities[choice], 4),
            "probabilities": {option: round(probabilities[option], 4) for option in options},
        }
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
        "confidence": round(max(probabilities[level] for level in levels), 4),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {level: round(probabilities[level], 4) for level in levels},
    }


def response(request: dict, questions, probs, input_tokens: int) -> dict:
    """The response body from the encoded questions (request.EncodedQuestion) and their float32 probabilities."""
    answers = {q.question_id: systemone_answer(request["questions"][q.question_id],
                                               dict(zip(q.option_ids, np.asarray(p, np.float32).tolist())))
               for q, p in zip(questions, probs)}
    return {"model": request["model"], "answers": answers,
            "usage": {"input_tokens": int(input_tokens), "output_tokens": 0}}
