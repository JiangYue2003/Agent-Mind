import sys
import types
import unittest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))
sys.modules.setdefault("redis", types.ModuleType("redis"))

from memory.conversation_memory import MemoryContext, Message, MsgRole


class MemoryContextPromptTests(unittest.TestCase):
    def test_prompt_can_exclude_cross_conversation_memory_from_final_answer(self):
        context = MemoryContext(
            recent_messages=[Message(role=MsgRole.USER, content="订单号是 ORD20250701001")],
            relevant_history=["旧会话中订单 ORD20250701003 已待发货。"],
            user_profile={"last_order": "ORD20250701003"},
            summary="当前会话摘要",
        )

        prompt = context.to_prompt_text(include_cross_conversation=False)

        self.assertIn("当前会话摘要", prompt)
        self.assertIn("ORD20250701001", prompt)
        self.assertNotIn("ORD20250701003", prompt)
        self.assertNotIn("[相关历史]", prompt)
        self.assertNotIn("[用户画像]", prompt)
