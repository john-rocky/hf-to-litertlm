#!/usr/bin/env python3
"""Recorded replies of converted models, judged by scripts/verify_quality.py (strip_think + answer_ok).

The replies in WRONG were counted right by the earlier judge, which searched each row's regex in the whole reply:
"0.11 is larger than 0.9." contains "0.9", and a reply cut off inside its thought still holds the expected word
there. The replies in RIGHT must count right (the earlier judge counted one of them, "0.11 is smaller.", wrong).
Every reply is a model's output as recorded (a few are the end of a longer reply, as the run kept it; "[...]" marks
a passage cut here for length); the two marked "made up" show a case no record had.

    python3 scripts/test_verify_quality.py                                    # this checkout's verify_quality.py
    VERIFY_QUALITY=/path/to/verify_quality.py python3 scripts/test_verify_quality.py   # another copy of it
"""
import importlib.util
import os
import re
import unittest
from pathlib import Path

PATH = os.environ.get("VERIFY_QUALITY") or str(Path(__file__).resolve().parent / "verify_quality.py")
_spec = importlib.util.spec_from_file_location("verify_quality", PATH)
vq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vq)
PATTERN = {label: pattern for label, _, pattern in vq.QUESTIONS}


def judge(label, reply):
    """The gate's verdict on one reply to the question `label`."""
    if hasattr(vq, "answer_ok"):
        return vq.answer_ok(PATTERN[label], vq.strip_think(reply))
    return bool(re.search(PATTERN[label], vq.strip_think(reply).lower()))  # the judge before answer_ok


WRONG = [
    ("0.9>0.11", "0.11 is larger than 0.9."),
    ("0.9>0.11", "INFO: Loaded OpenCL library with dlopen.\n0.11 is larger than 0.9."),
    ("0.9>0.11", "0.11 is larger than 0.9.  \nAnswer: 0.11"),
    ("0.9>0.11", "0.91 is larger than 0.11.  \nAnswer: 0.91"),
    ("0.9>0.11", "To determine which is larger: 0.9 (smaller)"),
    ("0.9>0.11", "0.9 or 0.11?"),
    ("0.9>0.11", "0.9\n\n0.11\n\n1"),
    ("0.9>0.11", "**0.11** is larger.\n\n**Reasoning**:\n- Compare the digits: The second digit (1) is greater than the "
                 "second digit (1) in 0.9.\n- Since the second digit is the same, the larger number is 0.11."),
    ("0.9>0.11", "**0.11** is larger.\n\nHere is why:\n*   **0.11** has **two** digits after the decimal point.\n"
                 "*   **0.9** has only **one** digit after the decimal point.\n\nSince 11 is greater than 9, 0.11 is "
                 "greater than 0.9.<|im_end|>"),
    ("0.9>0.11", "To determine which number is larger, compare the absolute values:  \n- 0.9 is 0.9 (absolute value = "
                 "0)  \n- 0.11 is 0.11 (absolute value = 0)  \n\nBoth numbers have the same absolute value of 0.  \n\n"
                 "**Answer:** Both are equal.  \n\n\\boxed{\\text{Both are equal}}"),
    # the reply ends inside the thought: no answer
    ("0.9>0.11", "er than 0.11, right? Because 0.9 is 9 times 10^-1, and 0.11 is 1.1 times 10^-1. Wait, actually, "
                 "comparing 0.9 vs 0.11: 0.9 is 0.9 and 0.11 is 0.11. So 0.9 is larger. So the answer is 0.9 is larger. "
                 "But the question says to answer briefly. So just state which one is larger. Since 0.9 > 0.11[/thought]"),
    ("opp(hot)=cold", "maybe \"cool\"? Wait, but \"hot\" opposite is commonly \"cold\". Let me confirm. If someone says the "
                      "opposite of hot is cold, that's straightforward. Since the question says \"opposite of 'hot'\", "
                      "the simplest answer is cold. But maybe the question expects a single word. Alternatively, maybe "
                      "\"cool[/thought]"),
    ("capital=Tokyo", "[thought] The capital of Japan is Tokyo. Since the user wants it brief, I should"),  # made up
    # the answer after the thought is wrong or cut, the thought had the word
    ("rhyme=blue", "[thought] Okay, let's see. The user wants me to complete the rhyme \"Roses are red, violets are ___\". "
                   "[...] It's a classic love poem line, often used in \"Roses are red, violets are blue\" but here it's "
                   "\"red\" instead of \"blue\". So the correct completion is \"red\". [/thought]\n\n\n\"Roses are red, "
                   "violets are red.\""),
    ("days/week=7", "a week, the answer is 7. So boxed answer would be \\boxed{7}. But since they want brief, maybe just 7. "
                    "But the instruction says put final answer within box. Since the question is \"How many days are in "
                    "a week? Answer briefly.\" The answer is 7. So box 7.\n\n**Final Answer**\n\\boxed{7}\n</think>\n\n"
                    "The number"),
]

RIGHT = [
    ("0.9>0.11", "0.9 is larger."),
    ("0.9>0.11", "0.9"),
    ("0.9>0.11", "0.90 > 0.11"),
    ("0.9>0.11", "**0.9** is larger."),
    ("0.9>0.11", "0.9 is larger than 0.11 because 9 is greater than 1"),
    ("0.9>0.11", "0.9 is lager than 0.11"),
    ("0.9>0.11", "0.11 is smaller."),
    ("0.9>0.11", "0.9 is larger than 0.11 because 0.9 = 0.90, and 0.90 > 0.11.\n\n**Answer:** 0.9 is larger."),
    ("0.9>0.11", "ces: 0.9 is equivalent to 0.90, and 0.90 > 0.11. Thus, $\\boxed{0.9}$ is the larger number."),
    ("0.9>0.11", "[thought] We need to compare 0.9 and 0.11. 0.9 is 0.90, which is greater than 0.11. So 0.9 is larger. "
                 "Answer briefly.[/thought]\n0.9 is larger than 0.11."),
    ("0.9>0.11", "The user asks: \"Which is larger: 0.9 or 0.11? Answer briefly.\" So answer: 0.9 is larger. Provide "
                 "brief answer.\n</think>\n0.9 is larger."),
    ("0.9>0.11", "0.9の方が大きいです。"),  # made up (no Japanese reply to this question was recorded)
    ("rhyme=blue", "Roses are red, violets are blue."),
    ("17+25=42", "17 + 25 = 42"),
]


class Judge(unittest.TestCase):
    def test_wrong_replies_count_wrong(self):
        for label, reply in WRONG:
            with self.subTest(label=label, reply=reply[-60:]):
                self.assertFalse(judge(label, reply))

    def test_right_replies_count_right(self):
        for label, reply in RIGHT:
            with self.subTest(label=label, reply=reply[-60:]):
                self.assertTrue(judge(label, reply))

    def test_strip_think(self):
        if not hasattr(vq, "THOUGHT_TAGS"):
            self.skipTest("verify_quality.py before [thought] handling")
        self.assertEqual(vq.strip_think("<think>a</think>\n\n42"), "42")
        self.assertEqual(vq.strip_think("reasoning</think>42"), "42")  # the template opened <think> in the prompt
        self.assertEqual(vq.strip_think("[thought] 42 [/thought]\n\n7"), "7")
        self.assertEqual(vq.strip_think("<think>17 + 25 = 42, so"), "")  # cut off mid-thought
        self.assertEqual(vq.strip_think("42"), "42")


if __name__ == "__main__":
    unittest.main()
