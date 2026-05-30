from pathlib import Path
import unittest


WEB_UI = Path(__file__).resolve().parents[1] / "web_ui.py"


class WebUIContractTests(unittest.TestCase):
    def test_chatbot_uses_messages_format(self):
        source = WEB_UI.read_text(encoding="utf-8")

        self.assertIn("chatbot = gr.Chatbot(height=420, show_label=False)", source)
        self.assertIn('{"role": "user", "content": message}', source)

    def test_gradio_6_places_css_and_theme_on_launch(self):
        source = WEB_UI.read_text(encoding="utf-8")

        self.assertIn("with gr.Blocks() as demo:", source)
        self.assertIn("css=CUSTOM_CSS", source)
        self.assertIn("theme=gr.themes.Monochrome()", source)

    def test_ui_has_explicit_send_button_bound_to_chat_handler(self):
        source = WEB_UI.read_text(encoding="utf-8")

        self.assertIn("send_btn = gr.Button", source)
        self.assertIn("send_btn.click(", source)
        self.assertIn("chat_fn", source)

    def test_create_session_updates_active_session_state_directly(self):
        source = WEB_UI.read_text(encoding="utf-8")

        self.assertIn("[session_dropdown, session_dropdown_bottom, init_status, chatbot, active_sid]", source)
        self.assertIn("def create_session_for_ui", source)
        self.assertIn("return gr.update(choices=choices, value=sid), gr.update(value=f\"{name} 已连接\"), [], sid", source)


if __name__ == "__main__":
    unittest.main()
