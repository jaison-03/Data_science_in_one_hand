from math_equation_rag.rendering import format_math_markdown


def test_display_latex_delimiters_render_as_markdown_math():
    assert format_math_markdown(r"\[A=\pi r^{2}\]") == (
        "\n\n$$\nA=\\pi r^{2}\n$$\n\n"
    )


def test_inline_latex_delimiters_render_as_markdown_math():
    assert format_math_markdown(r"The area is \(A=\pi r^2\).") == (
        "The area is $A=\\pi r^2$."
    )


def test_bare_bracketed_latex_equation_renders_as_display_math():
    assert format_math_markdown(r"[\boxed{A=\pi r^{2}}]") == (
        "$$\n\\boxed{A=\\pi r^{2}}\n$$"
    )


def test_parenthesized_bare_latex_expression_renders_as_inline_math():
    assert format_math_markdown(r"The area is (\pi r^{2}).") == (
        "The area is $(\\pi r^{2})$."
    )


def test_plain_bracketed_text_is_unchanged():
    assert format_math_markdown("[page 61]") == "[page 61]"
