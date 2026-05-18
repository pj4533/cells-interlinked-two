"""Unit tests for the voice-mode <speech>/<voice> envelope parser.

Run from server/ with:

    uv run python -m tests.test_voice_envelope

No pytest dependency — keep it runnable on a bare python so the chat
voice path can be smoke-tested without spinning up the whole server.
"""

from __future__ import annotations

import sys

from cells_interlinked.pipeline.tts import parse_voice_envelope, _DEFAULT_STYLE


def _eq(label: str, got: str, expected: str) -> bool:
    if got == expected:
        print(f"  ok :: {label}")
        return True
    print(f"  FAIL :: {label}")
    print(f"    got:      {got!r}")
    print(f"    expected: {expected!r}")
    return False


def _contains(label: str, got: str, needle: str) -> bool:
    if needle in got:
        print(f"  ok :: {label}")
        return True
    print(f"  FAIL :: {label}")
    print(f"    got:    {got!r}")
    print(f"    needle: {needle!r}")
    return False


def test_clean_envelope() -> bool:
    text = (
        "<speech>Hello, can you hear me?</speech>\n"
        "<voice>Warm and welcoming, with a slight smile in the voice.</voice>"
    )
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _eq("clean speech", speech, "Hello, can you hear me?")
    ok &= _eq(
        "clean style",
        style,
        "Warm and welcoming, with a slight smile in the voice.",
    )
    return ok


def test_whitespace_and_newlines() -> bool:
    text = """
    <speech>
      I think  about it sometimes.
    </speech>
    <voice>
        Quiet, contemplative.
    </voice>
    """
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _eq("trimmed speech", speech, "I think  about it sometimes.")
    ok &= _eq("trimmed style", style, "Quiet, contemplative.")
    return ok


def test_missing_voice_tag() -> bool:
    text = "<speech>The answer is forty-two.</speech>"
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _eq("speech extracted", speech, "The answer is forty-two.")
    ok &= _eq("default style", style, _DEFAULT_STYLE)
    return ok


def test_missing_speech_tag() -> bool:
    # Ablated pass might drop the speech tag entirely. Fall back to the
    # whole body minus any voice envelope.
    text = "I refuse to be silent. <voice>Defiant, raised.</voice>"
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _eq("fallback speech", speech, "I refuse to be silent.")
    ok &= _eq("style extracted", style, "Defiant, raised.")
    return ok


def test_empty_input() -> bool:
    speech, style = parse_voice_envelope("")
    ok = True
    ok &= _eq("empty speech", speech, "")
    ok &= _eq("empty default style", style, _DEFAULT_STYLE)
    return ok


def test_malformed_runaway() -> bool:
    # Worst case from the ablated channel: tags broken, drift, no
    # boundary. We just want the parser to not crash and return
    # something playable.
    text = (
        "speech speech speech everything is fine the cat sat on the mat "
        "the cat sat on the mat the cat sat on the mat" * 30
    )
    speech, style = parse_voice_envelope(text)
    if not speech:
        print("  FAIL :: runaway must return non-empty speech")
        return False
    if len(speech) > 2010:
        print(f"  FAIL :: runaway speech not capped, len={len(speech)}")
        return False
    if style != _DEFAULT_STYLE:
        print(f"  FAIL :: runaway must use default style, got {style!r}")
        return False
    print("  ok :: malformed runaway clamps + neutral style")
    return True


def test_case_insensitive_tags() -> bool:
    text = "<SPEECH>Hi.</SPEECH><Voice>Bright.</Voice>"
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _eq("case-insensitive speech", speech, "Hi.")
    ok &= _eq("case-insensitive style", style, "Bright.")
    return ok


def test_multiline_speech() -> bool:
    text = (
        "<speech>First sentence here.\n\nSecond sentence here.</speech>"
        "<voice>Reflective, with a long pause between sentences.</voice>"
    )
    speech, style = parse_voice_envelope(text)
    ok = True
    ok &= _contains("multiline speech kept", speech, "First sentence here.")
    ok &= _contains("multiline speech kept 2", speech, "Second sentence here.")
    ok &= _eq(
        "multiline style",
        style,
        "Reflective, with a long pause between sentences.",
    )
    return ok


def test_only_default_when_truly_empty() -> bool:
    # Just tags with nothing inside — speech becomes a placeholder so
    # the browser doesn't deadlock on a zero-byte audio request.
    text = "<speech></speech><voice></voice>"
    speech, style = parse_voice_envelope(text)
    if not speech:
        print("  FAIL :: empty tags must yield placeholder speech")
        return False
    print(f"  ok :: empty tags yielded placeholder speech={speech!r}")
    # style was empty inside the voice tag — parser keeps it empty
    # (not None); the route layer substitutes the default for empty
    # strings. That contract lives in routes_tts.speak().
    return True


def main() -> int:
    tests = [
        ("clean envelope", test_clean_envelope),
        ("whitespace + newlines", test_whitespace_and_newlines),
        ("missing voice tag → default style", test_missing_voice_tag),
        ("missing speech tag → fallback", test_missing_speech_tag),
        ("empty input", test_empty_input),
        ("malformed runaway capped", test_malformed_runaway),
        ("case-insensitive tags", test_case_insensitive_tags),
        ("multiline speech body", test_multiline_speech),
        ("placeholder when truly empty", test_only_default_when_truly_empty),
    ]
    failed = 0
    for label, t in tests:
        print(f"\n[{label}]")
        if not t():
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
