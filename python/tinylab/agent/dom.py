"""Describe candidate elements from Mind2Web's cleaned HTML: tag, key attributes, own text, parent text."""

from __future__ import annotations

from html.parser import HTMLParser

KEY_ATTRIBUTES = ("aria_label", "title", "alt", "placeholder", "name", "type", "role", "value", "input_value")
VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
OWN_WORDS = 24
PARENT_WORDS = 12
ATTRIBUTE_CHARS = 60
# Only the nearest ancestors collect text, so a page's root never accumulates the whole page.
TEXT_DEPTH = 8


class _Tree(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.nodes: dict[str, tuple[str, dict, list[str], str | None]] = {}
        self.stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag, attrs):
        attributes = {key: value for key, value in attrs if value}
        node_id = attributes.pop("backend_node_id", None)
        parent = next((ancestor for _, ancestor in reversed(self.stack) if ancestor is not None), None)
        if node_id is not None:
            self.nodes[node_id] = (tag, attributes, [], parent)
        if tag not in VOID_TAGS:
            self.stack.append((tag, node_id))

    def handle_endtag(self, tag):
        for depth in range(len(self.stack) - 1, -1, -1):
            if self.stack[depth][0] == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data):
        words = data.split()
        if not words:
            return
        collected = 0
        for _, node_id in reversed(self.stack):
            if node_id is None:
                continue
            bucket = self.nodes[node_id][2]
            room = OWN_WORDS - len(bucket)
            if room > 0:
                bucket.extend(words[:room])
            collected += 1
            if collected == TEXT_DEPTH:
                break


def describe(nodes: dict, node_id: str) -> str | None:
    if node_id not in nodes:
        return None
    tag, attributes, words, parent = nodes[node_id]
    parts = [tag]
    parts += [attributes[key][:ATTRIBUTE_CHARS] for key in KEY_ATTRIBUTES if key in attributes]
    parts += words
    if parent in nodes and nodes[parent][2]:
        parts += ["|", *nodes[parent][2][:PARENT_WORDS]]
    return " ".join(parts)


def describe_candidates(cleaned_html: str, node_ids: list[str]) -> dict[str, str]:
    """Description per candidate id; ids missing from the HTML are left out."""
    tree = _Tree()
    tree.feed(cleaned_html)
    tree.close()
    found = {node_id: describe(tree.nodes, node_id) for node_id in node_ids}
    return {node_id: text for node_id, text in found.items() if text is not None}
