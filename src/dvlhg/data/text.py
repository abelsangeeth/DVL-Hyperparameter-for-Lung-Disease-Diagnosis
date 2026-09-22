"""Radiology-report parsing, section selection and label redaction.

The text channel is the part of this project most easily got wrong. MIMIC-CXR's
CheXpert labels were produced by an NLP labeler that read FINDINGS and
IMPRESSION. Feeding those sections to the model and reporting the AUC is
circular: you are asking the model to re-read the sentence the label came from,
and you will see ~0.99 AUC that means nothing.

`build_text` therefore defaults to the *pre-read* sections only. The leaky modes
are still available, clearly named, so the ablation table can quantify the gap.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List

from ..constants import (
    DISEASE_TERMS,
    MASK_TOKEN,
    PRE_READ_SECTIONS,
    REPORT_SECTIONS,
)

# MIMIC de-identification replaces PHI with runs of underscores.
_DEID = re.compile(r"_{2,}")
_WS = re.compile(r"\s+")
_HEADER = re.compile(
    r"^[ \t]*(" + "|".join(sorted((re.escape(s) for s in REPORT_SECTIONS), key=len, reverse=True)) + r")[ \t]*:",
    re.IGNORECASE | re.MULTILINE,
)
_DISEASE = re.compile(r"\b(?:" + "|".join(DISEASE_TERMS) + r")\b", re.IGNORECASE)

TEXT_MODES = (
    "indication",
    "findings",
    "impression",
    "findings_masked",
    "full",
    "full_masked",
    "none",
)


def clean(text: str, lowercase: bool = True) -> str:
    """Collapse whitespace, drop de-identification underscores."""
    if not text:
        return ""
    text = _DEID.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WS.sub(" ", text).strip(" .;,-")
    return text.lower() if lowercase else text


def split_sections(report: str) -> Dict[str, str]:
    """Split a raw report into {UPPERCASE_HEADER: body}.

    Text before the first recognised header is stored under "PREAMBLE".
    Repeated headers are concatenated rather than overwritten.
    """
    if not report:
        return {}
    matches = list(_HEADER.finditer(report))
    sections: Dict[str, str] = {}
    if not matches:
        return {"PREAMBLE": report.strip()}
    if matches[0].start() > 0:
        preamble = report[: matches[0].start()].strip()
        if preamble:
            sections["PREAMBLE"] = preamble
    for i, match in enumerate(matches):
        name = match.group(1).upper().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        body = report[start:end].strip()
        if name in sections:
            sections[name] = f"{sections[name]} {body}".strip()
        else:
            sections[name] = body
    return sections


def mask_findings(text: str) -> str:
    """Redact the four target findings (and close confounders) from free text."""
    return _DISEASE.sub(MASK_TOKEN, text)


def _join(sections: Dict[str, str], names: Iterable[str]) -> str:
    parts: List[str] = []
    for name in names:
        body = sections.get(name, "").strip()
        if not body:
            continue
        # "FINAL REPORT" is a banner, not content.
        if name == "FINAL REPORT" and len(body) < 4:
            continue
        parts.append(f"{name.lower()}: {body}")
    return " ".join(parts)


def build_text(report: str, mode: str = "indication", lowercase: bool = True) -> str:
    """Turn a raw report into the string the text encoder will see.

    mode:
      indication      pre-read sections only (exam, indication, history,
                      technique, comparison) — the leakage-free default
      findings        FINDINGS only                      [leaky]
      impression      IMPRESSION/CONCLUSION only         [leaky]
      findings_masked FINDINGS with finding words redacted
      full            everything                         [leaky]
      full_masked     everything with finding words redacted
      none            empty string (image-only ablation)
    """
    if mode == "none":
        return ""
    if mode not in TEXT_MODES:
        raise ValueError(f"unknown text mode '{mode}'; expected one of {TEXT_MODES}")

    sections = split_sections(report or "")
    if mode == "indication":
        text = _join(sections, PRE_READ_SECTIONS)
        if not text and "PREAMBLE" in sections:
            # Some reports are a single unsectioned paragraph. Keeping it whole
            # would leak, so fall back to nothing rather than to the findings.
            text = ""
    elif mode in ("findings", "findings_masked"):
        text = _join(sections, ["FINDINGS"])
        if mode == "findings_masked":
            text = mask_findings(text)
    elif mode == "impression":
        text = _join(sections, ["IMPRESSION", "CONCLUSION"])
    else:  # full / full_masked
        ordered = [name for name in REPORT_SECTIONS if name in sections]
        text = _join(sections, ordered) or sections.get("PREAMBLE", "")
        if mode == "full_masked":
            text = mask_findings(text)

    return clean(text, lowercase=lowercase)


def is_leaky(mode: str) -> bool:
    """True when the text channel can contain the sentence the label came from."""
    return mode in {"findings", "impression", "full"}


def describe_mode(mode: str) -> str:
    return {
        "indication": "pre-read clinical context only (leakage-free)",
        "findings": "FINDINGS section (LEAKY: labels were derived from it)",
        "impression": "IMPRESSION section (LEAKY: labels were derived from it)",
        "findings_masked": "FINDINGS with finding terms redacted (partially leaky)",
        "full": "whole report (LEAKY)",
        "full_masked": "whole report with finding terms redacted (partially leaky)",
        "none": "no text - image-only ablation",
    }.get(mode, mode)
