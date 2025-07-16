def chunk_text(text: str, max_tokens: int = 256) -> list[str]:
    import textwrap
    return textwrap.wrap(text, max_tokens)