#!/usr/bin/env python3
"""DEPRECATED: the historical annotator modified v1 sequence files in place.

Raw teacher continuations and their existing annotations are preserved. Phase 7
will perform the required manual review into a separate corrected-v2 artifact;
Phase 1 must not infer or write review decisions.
"""
from __future__ import annotations


DEPRECATION_MESSAGE = (
    "Refusing in-place degeneration annotation: v1 artifacts are immutable. "
    "Use the future Phase 7 corrected-v2 review workflow."
)


def main() -> None:
    raise RuntimeError(DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()
