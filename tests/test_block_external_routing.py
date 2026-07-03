"""Manual tests for the block_external_routing validation in LLMSettings."""

from tester.config import LLMSettings


def test():
    passed = 0
    failed = 0

    cases = [
        # (kwargs, should_pass, description)
        (
            {"block_external_routing": True},
            False,
            "default api_base is openai.com (should reject when blocked)",
        ),
        (
            {"api_base": "http://localhost:11434/v1", "block_external_routing": True},
            True,
            "localhost accepted when blocked",
        ),
        (
            {"api_base": "http://127.0.0.1:8000/v1", "block_external_routing": True},
            True,
            "127.0.0.1 accepted when blocked",
        ),
        (
            {"api_base": "http://192.168.1.100:11434/v1", "block_external_routing": True},
            True,
            "192.168.x.x private IP accepted when blocked",
        ),
        (
            {"api_base": "http://10.0.0.1:8000/v1", "block_external_routing": True},
            True,
            "10.x.x.x private IP accepted when blocked",
        ),
        (
            {"api_base": "https://api.openai.com/v1", "block_external_routing": True},
            False,
            "api.openai.com rejected when blocked",
        ),
        (
            {"api_base": "https://api.openai.com/v1", "block_external_routing": False},
            True,
            "api.openai.com accepted when not blocked (default)",
        ),
    ]

    for kwargs, should_pass, desc in cases:
        try:
            LLMSettings(**kwargs)
            actual_pass = True
        except Exception as e:
            actual_pass = False

        status = "PASS" if actual_pass == should_pass else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {desc}")
        if status == "FAIL":
            print(f"       expected pass={should_pass}, got pass={actual_pass}")

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    test()