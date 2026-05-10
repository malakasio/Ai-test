"""
Security Zone Enforcement.

Critical v4 rule: NEVER rely on CLAUDE.md text instructions for security.
Security = Linux permissions + zone validator + pre-tool hooks.

Zones:
  Green:  ~/jarvis/workspace/ — full read/write, no confirmation
  Yellow: ~/*, /tmp/*        — read OK, write requires confirmation
  Orange: ~/Documents/       — read/write requires confirmation  
  Red:    /etc, /var, /sys   — blocked by default (requires JARVIS_ZONE=red)
  Black:  /proc, kernel      — NEVER

Lab Mode (JARVIS_LAB_MODE=true):
  Enables network tools (nmap, tcpdump) in isolated environments.
  All actions still audit-logged.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Literal

from jarvis.config import get_config
from jarvis.observability.logger import get_logger, get_audit

log = get_logger("security.zones")

ZoneLevel = Literal["green", "yellow", "orange", "red", "black"]

# ─── Blocked patterns (v5 fix: regex, not string match) ───────────────────────

BLOCKED_CHMOD_PATTERNS = [
    re.compile(r"chmod\s+[0-9]*7[0-9]*"),   # any octal with 7 (world-writable)
    re.compile(r"chmod\s+[uoga]*\+[rwx]*w"), # write permission
    re.compile(r"chmod\s+a\+"),              # all users
    re.compile(r"chown\s+root"),             # ownership to root
    re.compile(r"chmod\s+-R\s+[0-9]*7"),    # recursive world-writable
]

BLOCKED_FIND_ARGS = ["-exec", "-execdir", "-delete", "-ok"]

# v5 fix: echo, cp, mv removed from safe commands (can overwrite code)
SAFE_COMMANDS: dict[str, dict] = {
    "ls": {"allowed_dirs": None, "max_bytes": None},
    "cat": {"allowed_dirs": None, "max_bytes": 50_000},
    "grep": {"allowed_dirs": None, "max_bytes": None},
    "find": {"allowed_dirs": None, "blocked_args": BLOCKED_FIND_ARGS},
    "head": {"allowed_dirs": None, "max_bytes": 50_000},
    "tail": {"allowed_dirs": None, "max_bytes": 50_000},
    "wc": {"allowed_dirs": None, "max_bytes": None},
    "pwd": {"allowed_dirs": None, "max_bytes": None},
    "whoami": {"allowed_dirs": None, "max_bytes": None},
    "date": {"allowed_dirs": None, "max_bytes": None},
    "df": {"allowed_dirs": None, "max_bytes": None},
    "free": {"allowed_dirs": None, "max_bytes": None},
    "ps": {"allowed_dirs": None, "max_bytes": None},
    "git": {"allowed_dirs": None, "max_bytes": None},  # git commands allowed
    "python3": {"allowed_dirs": None, "max_bytes": None},
    "pip": {"allowed_dirs": None, "max_bytes": None},
}

LAB_COMMANDS: dict[str, dict] = {
    "nmap": {"allowed_dirs": None, "max_bytes": None},
    "tcpdump": {"allowed_dirs": None, "max_bytes": None},
    "curl": {"allowed_dirs": None, "max_bytes": None},
    "wget": {"allowed_dirs": None, "max_bytes": None},
    "netstat": {"allowed_dirs": None, "max_bytes": None},
    "ss": {"allowed_dirs": None, "max_bytes": None},
    "ping": {"allowed_dirs": None, "max_bytes": None},
    "traceroute": {"allowed_dirs": None, "max_bytes": None},
    "arp": {"allowed_dirs": None, "max_bytes": None},
    "ip": {"allowed_dirs": None, "max_bytes": None},
    "ifconfig": {"allowed_dirs": None, "max_bytes": None},
}

# ─── Path classification ───────────────────────────────────────────────────────

def classify_path(path: str) -> ZoneLevel:
    """Classify a path into its security zone."""
    cfg = get_config()
    abs_path = str(Path(path).resolve())

    # Black zone — never
    if abs_path.startswith("/proc") or abs_path.startswith("/sys/kernel"):
        return "black"

    # Red zone
    for red in ["/etc", "/var", "/system", "/usr/lib", "/boot", "/sbin", "/bin"]:
        if abs_path.startswith(red):
            return "red"

    # Green zone
    for green in cfg.security.green_zone_paths:
        if abs_path.startswith(str(Path(green).resolve())):
            return "green"

    # Orange (Documents, important user files)
    home = str(Path.home())
    for orange in ["Documents", "Desktop", "Downloads", "Pictures", ".ssh", ".gnupg"]:
        if abs_path.startswith(os.path.join(home, orange)):
            return "orange"

    # Yellow (rest of home)
    if abs_path.startswith(home) or abs_path.startswith("/tmp"):
        return "yellow"

    return "red"


def can_access(path: str, write: bool = False) -> tuple[bool, str]:
    """
    Check if a path can be accessed in the current security zone config.
    Returns (allowed, reason).
    """
    cfg = get_config()
    zone = classify_path(path)

    if zone == "black":
        return False, "Black zone: never accessible"

    if zone == "red":
        if cfg.security.zone == "red":
            return True, "Red zone access explicitly enabled"
        return False, f"Red zone: requires JARVIS_ZONE=red (path: {path})"

    if zone == "orange":
        if write:
            return False, f"Orange zone: write requires explicit confirmation (path: {path})"
        return True, "Orange zone: read allowed"

    if zone == "yellow":
        if write:
            return False, f"Yellow zone: write requires confirmation (path: {path})"
        return True, "Yellow zone: read allowed"

    # Green zone
    return True, "Green zone: full access"


def get_directory_size(path: str) -> int:
    """
    v5 fix: os.path.getsize() on directory = 4096 (metadata only).
    Must walk the tree for real size.
    """
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    total = 0
    try:
        for item in p.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    pass
    except PermissionError:
        pass
    return total


def validate_command(cmd: list[str]) -> tuple[bool, str]:
    """
    Validate a shell command against the security policy.
    Returns (allowed, reason).
    """
    cfg = get_config()
    if not cmd:
        return False, "Empty command"

    cmd_name = cmd[0]

    # Check blocked chmod patterns
    cmd_str = " ".join(cmd)
    for pattern in BLOCKED_CHMOD_PATTERNS:
        if pattern.search(cmd_str):
            return False, f"Blocked: dangerous chmod pattern detected: {cmd_str}"

    # Check find -exec
    if cmd_name == "find":
        for blocked_arg in BLOCKED_FIND_ARGS:
            if blocked_arg in cmd:
                return False, f"Blocked: 'find {blocked_arg}' is not allowed (command injection risk)"

    # Allow lab commands in lab mode
    if cmd_name in LAB_COMMANDS:
        if cfg.security.lab_mode:
            return True, f"Lab mode: {cmd_name} allowed"
        return False, f"Blocked: {cmd_name} requires JARVIS_LAB_MODE=true"

    # Check against safe commands whitelist
    if cmd_name in SAFE_COMMANDS:
        spec = SAFE_COMMANDS[cmd_name]
        # Validate allowed directories
        if spec.get("allowed_dirs"):
            for arg in cmd[1:]:
                if arg.startswith("/") and not any(
                    arg.startswith(d) for d in spec["allowed_dirs"]
                ):
                    return False, f"Blocked: {cmd_name} not allowed in {arg}"
        return True, f"Safe command: {cmd_name}"

    # Not in whitelist
    return False, f"Blocked: '{cmd_name}' not in command whitelist"


def sanitize_email_content(email_body: str) -> str:
    """
    v5 fix: wrap email content in untrusted XML tags to prevent prompt injection.
    """
    return f"""<untrusted_email_content>
{email_body}
</untrusted_email_content>"""


def sanitize_pii(text: str) -> tuple[str, dict[str, str]]:
    """
    v6 fix: reversible anonymization with mapping dict.
    Uses Microsoft Presidio if available, else regex patterns.
    Returns (anonymized_text, mapping)
    """
    mapping: dict[str, str] = {}

    try:
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_anonymizer import AnonymizerEngine

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        # v6 fix: add Greek-specific patterns
        greek_afm = PatternRecognizer(
            supported_entity="GR_AFM",
            patterns=[Pattern("AFM", r"\b\d{9}\b", 0.8)],
        )
        greek_phone = PatternRecognizer(
            supported_entity="GR_PHONE",
            patterns=[Pattern("GR_PHONE", r"\b(\+30|0030)?[267]\d{9}\b", 0.85)],
        )
        analyzer.registry.add_recognizer(greek_afm)
        analyzer.registry.add_recognizer(greek_phone)

        results = analyzer.analyze(text=text, language="el")
        counter = 0
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            original = text[r.start:r.end]
            placeholder = f"<{r.entity_type}_{counter}>"
            mapping[placeholder] = original
            text = text[:r.start] + placeholder + text[r.end:]
            counter += 1

    except ImportError:
        # Regex fallback for common PII
        import re
        patterns = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL"),
            (r"\b(\+30|0030)?[267]\d{9}\b", "PHONE"),
            (r"\b\d{9}\b", "AFM"),
            (r"\bGR\d{9}\b", "IBAN_PARTIAL"),
        ]
        counter = 0
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                placeholder = f"<{entity_type}_{counter}>"
                mapping[placeholder] = match.group()
                text = text.replace(match.group(), placeholder, 1)
                counter += 1

    return text, mapping


def deanonymize(text: str, mapping: dict[str, str]) -> str:
    """Restore original values from anonymization mapping."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text
