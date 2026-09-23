from dataclasses import dataclass


@dataclass
class Q:
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"; image: bytes = None


NEUTRAL_NONE = "not listed here"


def neutralize_options(options):
    """The training augmentation used the literal 'none of the above', and the model learned that exact string as an
    abstain signal (it abstains even on clear cases when the string is offered). Any option that reads like it is
    rewritten to a neutral phrasing for the model and mapped back in the output."""
    out, back = [], {}
    for o in options:
        key = o.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a", "none of these"):
            out.append(NEUTRAL_NONE); back[NEUTRAL_NONE] = o
        else:
            out.append(o)
    return out, back

