"""Tests for prompt generation."""

from tester.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT_TEMPLATE,
    make_recap_system_prompt,
    make_recap_user_prompt,
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


class TestRecapPrompts:
    def test_recap_system_prompt_includes_experience_guidance(self):
        prompt = make_recap_system_prompt()
        # Still surfaces the developer takeaway fields.
        assert "next_action" in prompt
        assert "related_steps" in prompt
        # Now drives a persona-based, in-the-player's-shoes experience review.
        assert "persona_reviews" in prompt
        assert "overall_verdict" in prompt
        assert "Typical player" in prompt
        assert "penalize" in prompt.lower()

    def test_recap_system_prompt_custom_override(self):
        custom = "Summarize this run."
        result = make_recap_system_prompt(custom)
        assert result == "Summarize this run."

    def test_recap_user_prompt_includes_metadata(self):
        result = make_recap_user_prompt(
            steps_completed=10,
            completion_reason="completed",
            duration=42.5,
            action_counts="click=5, wait=3, done=1",
            step_log="Step 1 [click]: Started game\n  Reasoning: begin",
        )
        assert "10" in result
        assert "completed" in result
        assert "42.5" in result
        assert "click=5, wait=3, done=1" in result
        assert "Step 1 [click]: Started game" in result

    def test_recap_user_prompt_asks_for_json(self):
        result = make_recap_user_prompt(
            steps_completed=1,
            completion_reason="max_steps_reached",
            duration=1.0,
            action_counts="none",
            step_log="Step 1 [wait]: Nothing\n  Reasoning: idle",
        )
        assert "JSON" in result

    def test_recap_user_prompt_renders_configured_personas(self):
        result = make_recap_user_prompt(
            steps_completed=1,
            completion_reason="completed",
            duration=1.0,
            action_counts="none",
            step_log="Step 1 [done]: end\n  Reasoning: fin",
            personas=["Speedrunner", "Lore reader"],
        )
        assert "1. Speedrunner" in result
        assert "2. Lore reader" in result

    def test_recap_user_prompt_falls_back_to_typical_player(self):
        result = make_recap_user_prompt(
            steps_completed=1,
            completion_reason="completed",
            duration=1.0,
            action_counts="none",
            step_log="Step 1 [done]: end\n  Reasoning: fin",
            personas=[],
        )
        assert "Typical player" in result
