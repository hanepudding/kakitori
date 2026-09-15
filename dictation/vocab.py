from pathlib import Path


def load_prompt(path: Path) -> str | None:
    """Read one term per line (# starts a comment) into the biasing prompt Qwen3-ASR reads as a system message."""
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(line)
    terms = list(dict.fromkeys(terms))
    if not terms:
        return None
    return "Vocabulary: " + "、".join(terms) + "。"
