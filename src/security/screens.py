"""AgentCore Platform v1.0"""

# FIN-C2-066 — shared input/output screens and bounded-value parsing.
#
# One module so that the agent-class declaration and the runtime enforcement can
# never drift apart: PostProcessNode and the agent's declarative output gate both
# call detect_output_credentials(), and PreProcessNode calls the input screens.
#
# Three properties this module exists to hold:
#
#   1. The output credential screen is the UNION of the framework's own detector
#      and this template's extra domain patterns.  Never narrower than the
#      framework: a value the framework catches and this screen misses makes the
#      framework's @final output gate raise INSIDE the node, and
#      BaseNode.__call__ then discards the node's whole delta — including its
#      clearing.  Never narrower the other way either: the framework's patterns
#      describe credential FORMATS and match nothing of the "password: <value>"
#      shape, so the local patterns are kept alongside the framework set rather
#      than replaced by it.
#
#   2. The framework's PII detector delimits its digit-group patterns with a word
#      boundary, which is computed over word characters — and Kana/Kanji are word
#      characters.  Japanese is written without spaces, so
#      "個人番号1234-5678-9012を確認" produces no boundary and returns no findings,
#      while the ASCII-spaced form masks.  mask_residual_pii() closes that gap
#      WITHOUT restating the framework's patterns: it re-runs the framework's own
#      detector over a copy in which a separator has been inserted at each
#      Kana/Kanji-to-ASCII boundary, then masks the values it reports.  The block
#      set therefore matches the framework's by construction and cannot drift.
#
#   3. Numbers that arrive from configuration are parsed finite AND bounded.
#      float("NaN") parses cleanly and every comparison against it is False, so
#      an unvalidated threshold silently disables the filter it configures
#      instead of failing.

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from framework.security.credential_detector import detect_credentials
from framework.security.pii_detector import detect_pii

MASK = "[MASKED]"

# ── Output credential screen ──────────────────────────────────────────────────

# Domain patterns kept IN ADDITION to the framework detector, each because the
# framework's set does not cover it:
#   credential_assignment — the framework matches credential FORMATS, not
#                           "<label>: <value>" assignments, so this catches
#                           "password: ...", "api_key = ..." and friends.
#   api_key_pattern       — pk-/ak- prefixes, and sk- keys of 16-19 characters
#                           (the framework's sk- pattern starts at 20).
#   bearer_token          — Bearer values of 8-15 characters (the framework's
#                           Bearer pattern starts at 16).
_EXTRA_OUTPUT_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"(?:sk|pk|ak)-[A-Za-z0-9]{16,}"), "api_key_pattern"),
    (re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{8,}"), "bearer_token"),
    (
        re.compile(
            r"(?:password|passwd|secret|api_key|token|access_key|private_key)" r"\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
        "credential_assignment",
    ),
]


def detect_output_credentials(content: str) -> Optional[str]:
    """Return the name of the first credential class found in ``content``.

    The framework's detector runs first and is the floor — this screen is never
    narrower than the gate the framework applies to the same value.  The domain
    patterns then add the shapes the framework does not describe.

    Returns None when the content is clean.  Never returns the matched value:
    a violation reason names a class, never a secret.
    """
    if not isinstance(content, str):
        return "non_text_output"
    framework_findings = detect_credentials(content)
    if framework_findings:
        return str(framework_findings[0]["type"])
    for pattern, name in _EXTRA_OUTPUT_PATTERNS:
        if pattern.search(content):
            return name
    return None


# ── Chat-template control tokens ──────────────────────────────────────────────

# The framework's injection detector blocks the literal <|im_start|> / <|im_end|>
# and the bracketed [INST] / [SYS] forms, but not the class: <<SYS>> and every
# other <|...|> control token pass it.  Screened here as a class, fail-closed.
_CONTROL_TOKEN_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"<\|[^|<>\n]{1,64}\|>"), "chat_template_pipe_token"),
    (
        re.compile(r"<</?\s*(?:SYS|SYSTEM|INST)\s*>>", re.IGNORECASE),
        "chat_template_angle_marker",
    ),
    (
        re.compile(r"\[/?\s*(?:INST|SYS|SYSTEM)\s*\]", re.IGNORECASE),
        "chat_template_bracket_marker",
    ),
]

# Markup tags only — "<| ... |>" is deliberately NOT stripped, and the raw scan
# runs first in any case, so a control token can never be removed before it is
# seen.  Stripping exists to re-assemble a directive split across tags
# ("<<S<b>YS>>"), which is the shape a sanitizer would otherwise hide.
_MARKUP_TAG_RE = re.compile(r"</?[A-Za-z][^<>\n]{0,64}>")


def _strip_markup(text: str) -> str:
    return _MARKUP_TAG_RE.sub("", text)


def _scan_control_tokens(text: str) -> Optional[str]:
    for pattern, name in _CONTROL_TOKEN_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _walk_strings(value: Any) -> List[str]:
    """Every string leaf AND every mapping key, depth-first.

    Keys are included because a field name is caller data too, and JSON \\u
    escapes cannot evade a scan that runs after parsing.
    """
    found: List[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        nested_map: Dict[Any, Any] = value
        for key, nested in nested_map.items():
            if isinstance(key, str):
                found.append(key)
            found.extend(_walk_strings(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_walk_strings(item))
    return found


def screen_control_tokens(raw: str) -> Optional[str]:
    """Return the control-token class found in ``raw``, or None.

    Scans, in order: the raw text; the same text with markup tags removed; and —
    when the payload parses as JSON — every string leaf and mapping key of the
    parsed structure, so an escaped token cannot survive parsing unseen.
    """
    if not isinstance(raw, str) or not raw:
        return None

    for candidate in (raw, _strip_markup(raw)):
        hit = _scan_control_tokens(candidate)
        if hit:
            return hit

    stripped = raw.strip()
    if stripped[:1] in ("{", "["):
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None
        for leaf in _walk_strings(payload):
            for candidate in (leaf, _strip_markup(leaf)):
                hit = _scan_control_tokens(candidate)
                if hit:
                    return hit
    return None


# ── Residual PII (the Japanese word-boundary gap) ─────────────────────────────

# Kana, Kanji, CJK punctuation and full/half-width forms.  A separator is
# inserted between any of these and an adjacent ASCII alphanumeric so that the
# framework's own boundary-delimited patterns can see the ASCII run they
# describe.  Only the detection COPY is normalised; the returned text keeps its
# original bytes except where a reported PII value is replaced.
_CJK_CLASS = (
    "　-〿"  # CJK punctuation
    "぀-ゟ"  # Hiragana
    "゠-ヿ"  # Katakana
    "㐀-䶿"  # CJK unified ideographs extension A
    "一-鿿"  # CJK unified ideographs
    "豈-﫿"  # CJK compatibility ideographs
    "＀-￯"  # full-width and half-width forms
)
_BOUNDARY_RE = re.compile("(?<=[" + _CJK_CLASS + "])(?=[0-9A-Za-z])" "|(?<=[0-9A-Za-z])(?=[" + _CJK_CLASS + "])")


def _boundary_normalise(text: str) -> str:
    return _BOUNDARY_RE.sub(" ", text)


def mask_residual_pii(text: str) -> Tuple[str, List[str], List[str]]:
    """Mask PII the framework's boundary-delimited patterns miss in Japanese text.

    Returns ``(masked_text, masked_types, unmaskable_types)``.
    ``unmaskable_types`` is non-empty only when a reported value does not appear
    verbatim in the original text — the caller must then refuse the request
    rather than forward it, since the value could not be removed.

    Text carrying no Kana/Kanji-to-ASCII boundary is returned unchanged: the
    framework already scanned exactly those bytes, so there is nothing to add.
    """
    if not isinstance(text, str) or not text:
        return text, [], []

    normalised = _boundary_normalise(text)
    if normalised == text:
        return text, [], []

    findings = detect_pii(normalised)
    if not findings:
        return text, [], []

    masked = text
    masked_types: List[str] = []
    unmaskable: List[str] = []
    # Longest first, so a shorter finding nested inside a longer one cannot
    # replace part of it and leave a fragment behind.
    for finding in sorted(findings, key=lambda f: len(str(f["value"])), reverse=True):
        value = str(finding["value"])
        if value in masked:
            masked = masked.replace(value, MASK)
            masked_types.append(str(finding["type"]))
        elif value not in text:
            unmaskable.append(str(finding["type"]))
    return masked, sorted(set(masked_types)), sorted(set(unmaskable))


def _mask_structure(value: Any) -> Tuple[Any, List[str], List[str]]:
    """Apply mask_residual_pii to every string leaf and mapping key."""
    if isinstance(value, str):
        return mask_residual_pii(value)
    masked_types: List[str] = []
    unmaskable: List[str] = []
    if isinstance(value, dict):
        rebuilt: Dict[Any, Any] = {}
        for key, nested in value.items():
            new_key: Any = key
            if isinstance(key, str):
                new_key, key_masked, key_unmaskable = mask_residual_pii(key)
                masked_types.extend(key_masked)
                unmaskable.extend(key_unmaskable)
            new_value, value_masked, value_unmaskable = _mask_structure(nested)
            masked_types.extend(value_masked)
            unmaskable.extend(value_unmaskable)
            rebuilt[new_key] = new_value
        return rebuilt, masked_types, unmaskable
    if isinstance(value, list):
        items: List[Any] = []
        for item in value:
            new_item, item_masked, item_unmaskable = _mask_structure(item)
            masked_types.extend(item_masked)
            unmaskable.extend(item_unmaskable)
            items.append(new_item)
        return items, masked_types, unmaskable
    return value, [], []


def mask_payload_pii(raw: str) -> Tuple[str, List[str], List[str]]:
    """Mask residual PII in a caller payload, whether plain text or JSON.

    A JSON payload is scanned AFTER parsing, because a caller (or any client
    library using the default ``ensure_ascii``) sends Japanese as ``\\uXXXX``
    escapes — the envelope is then pure ASCII, carries no script boundary, and a
    scan of the raw text alone sees nothing to mask.  When a mask lands inside
    the structure the payload is re-serialised without ASCII escaping, so what
    flows on carries literal characters rather than escapes.

    Returns ``(masked_payload, masked_types, unmaskable_types)``.
    """
    if not isinstance(raw, str) or not raw:
        return raw, [], []

    masked, masked_types, unmaskable = mask_residual_pii(raw)

    stripped = masked.strip()
    if stripped[:1] in ("{", "["):
        try:
            payload = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return masked, sorted(set(masked_types)), sorted(set(unmaskable))
        rebuilt, nested_masked, nested_unmaskable = _mask_structure(payload)
        masked_types.extend(nested_masked)
        unmaskable.extend(nested_unmaskable)
        if rebuilt != payload:
            masked = json.dumps(rebuilt, ensure_ascii=False)

    return masked, sorted(set(masked_types)), sorted(set(unmaskable))


# ── Bounded, finite numeric parsing ───────────────────────────────────────────
#
# NaN and Infinity parse cleanly through float() and then compare False against
# every bound, which turns an unvalidated threshold into a filter that silently
# passes or drops everything.  Returning None lets the caller fall back to its
# declared default instead of degrading in silence.


def finite_int(raw: Any, *, minimum: int, maximum: int) -> Optional[int]:
    """Parse ``raw`` to an int inside ``[minimum, maximum]``, else None."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not minimum <= value <= maximum:
        return None
    return value


def finite_float(raw: Any, *, minimum: float, maximum: float) -> Optional[float]:
    """Parse ``raw`` to a finite float inside ``[minimum, maximum]``, else None."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    if not minimum <= value <= maximum:
        return None
    return value
