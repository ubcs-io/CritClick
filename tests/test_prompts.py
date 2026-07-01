"""Tests for prompt generation."""

from tester.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    make_system_prompt,
    make_user_prompt,
)


class TestSystemPrompt:
    def test_default_includes_all_actions(self):
        assert '"click"' in DEFAULT_SYSTEM_PROMPT
        assert '"wait"' in DEFAULT_SYSTEM_PROMPT
        assert '"type"' in DEFAULT_SYSTEM_PROMPT
        assert '"press"' in DEFAULT_SYSTEM_PROMPT
        assert '"done"' in DEFAULT_SYSTEM_PROMPT

    def test_make_default(self):
        result = make_system_prompt()
        assert result == DEFAULT_SYSTEM_PROMPT.strip()

    def test_make_custom_override(self):
        result = make_system_prompt("Custom prompt")
        assert result == "Custom prompt"

    def test_press_describes_key_to_press(self):
        assert "key_to_press" in DEFAULT_SYSTEM_PROMPT


class TestUserPrompt:
    def test_default_template_has_context_placeholder(self):
        assert "{context}" in DEFAULT_USER_PROMPT_TEMPLATE

    def test_make_user_prompt_fills_context(self):
        result = make_user_prompt("The player is at the main menu.")
        assert "The player is at the main menu." in result

    def test_make_user_prompt_custom_template(self):
        custom = "Context: {context}"
        result = make_user_prompt("Test context", template=custom)
        assert result == "Context: Test context"
