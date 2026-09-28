# Excerpt of decider/infer.py from the Mapika/decider-2b-vision checkpoint (revision
# 863e290863655f1d6b69324d77d09ac972d21609, file SHA-256 a7bd1b5568bdf30d5804bfe9f17a2864224540845072c9ce9f25d647b05449c7).
# Changes: only the Q and Example dataclasses are kept, byte for byte; everything else (the torch-based Decider class
# and its imports) is removed so that the reference readout does not depend on torch.
from dataclasses import dataclass


@dataclass
class Q:
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"; image: bytes = None
