#!/usr/bin/env python3
from pathlib import Path
here = Path(__file__).resolve().parent
parts = sorted(here.glob("optimization.py.part*"))
assert parts, "missing parts"
text = "".join(p.read_text() for p in parts)
(here / "optimization.py").write_text(text)
print("Assembled optimization.py", len(text), "chars from", len(parts), "parts")
