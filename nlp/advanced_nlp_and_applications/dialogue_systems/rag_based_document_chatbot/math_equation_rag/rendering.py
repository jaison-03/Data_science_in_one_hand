from __future__ import annotations

import re

_DISPLAY_DELIMITERS = re.compile(r"\\\[([\s\S]*?)\\\]")
_INLINE_DELIMITERS = re.compile(r"\\\(([\s\S]*?)\\\)")
_BRACKETED_EQUATION = re.compile(
    r"(?m)^(?P<indent>[ \t]*)\[\s*"
    r"(?P<equation>[^\]\r\n]*\\[A-Za-z]+[^\]\r\n]*)\s*\]"
    r"(?P<trailing>[ \t]*)$"
)
_PARENTHESIZED_EQUATION = re.compile(
    r"(?<![$\\])\(\s*(?P<equation>[^()\r\n]*\\[A-Za-z]+[^()\r\n]*)\s*\)(?![$])"
)


def format_math_markdown(text: str) -> str:
    """Normalize common LaTeX delimiters for Streamlit's Markdown renderer."""
    formatted = _DISPLAY_DELIMITERS.sub(
        lambda match: f"\n\n$$\n{match.group(1).strip()}\n$$\n\n", text
    )
    formatted = _INLINE_DELIMITERS.sub(
        lambda match: f"${match.group(1).strip()}$", formatted
    )
    formatted = _BRACKETED_EQUATION.sub(
        lambda match: (
            f"{match.group('indent')}$$\n"
            f"{match.group('equation').strip()}\n$$"
            f"{match.group('trailing')}"
        ),
        formatted,
    )
    return _PARENTHESIZED_EQUATION.sub(
        lambda match: f"$({match.group('equation').strip()})$", formatted
    )
