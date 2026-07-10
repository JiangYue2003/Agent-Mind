import importlib.util
import pathlib
import unittest

import httpx
from fastapi.testclient import TestClient

from api import main as api_main


def _load_tool_manager_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "mcp" / "tool_manager.py"
    spec = importlib.util.spec_from_file_location("tool_manager_customer_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool_manager_module = _load_tool_manager_module()
MCPToolManager = tool_manager_module.MCPToolManager
Tool = tool_manager_module.Tool


class CustomerServiceToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_order_lookup_calls_mock_external_order_service(self):
        from mcp.order_lookup import OrderLookupService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = OrderLookupService(base_url=str(client.base_url), transport=transport)

        result = await service.lookup_handler(
            {"user_id": "u1001", "order_id": "ORD20250701001"},
            context=None,
        )

        self.assertEqual(result["order_id"], "ORD20250701001")
        self.assertEqual(result["user_id"], "u1001")
        self.assertEqual(result["status"], "运输中")
        self.assertEqual(result["source"], "mock_oms")

    async def test_order_lookup_hides_other_users_order(self):
        from mcp.order_lookup import OrderLookupService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = OrderLookupService(base_url=str(client.base_url), transport=transport)

        with self.assertRaises(httpx.HTTPStatusError) as ctx:
            await service.lookup_handler(
                {"user_id": "u999", "order_id": "ORD20250701001"},
                context=None,
            )

        self.assertEqual(ctx.exception.response.status_code, 404)

    async def test_shipment_track_calls_mock_external_logistics_service(self):
        from mcp.shipment_track import ShipmentTrackService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = ShipmentTrackService(base_url=str(client.base_url), transport=transport)

        result = await service.track_handler(
            {"user_id": "u1001", "order_id": "ORD20250701001"},
            context=None,
        )

        self.assertEqual(result["order_id"], "ORD20250701001")
        self.assertEqual(result["carrier"], "顺丰速运")
        self.assertGreaterEqual(len(result["events"]), 1)

    async def test_refund_create_requires_order_ownership(self):
        from mcp.refund_create import RefundCreateService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = RefundCreateService(base_url=str(client.base_url), transport=transport)

        with self.assertRaises(httpx.HTTPStatusError) as ctx:
            await service.create_handler(
                {"user_id": "u999", "order_id": "ORD20250701001", "reason": "买错了"},
                context=None,
            )

        self.assertEqual(ctx.exception.response.status_code, 404)

    async def test_refund_create_posts_to_mock_external_refund_service(self):
        from mcp.refund_create import RefundCreateService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = RefundCreateService(base_url=str(client.base_url), transport=transport)

        result = await service.create_handler(
            {"user_id": "u1001", "order_id": "ORD20250701001", "reason": "买错了"},
            context=None,
        )

        self.assertEqual(result["order_id"], "ORD20250701001")
        self.assertEqual(result["status"], "submitted")
        self.assertTrue(result["refund_id"].startswith("RF"))

    async def test_human_handoff_posts_context_to_mock_external_system(self):
        from mcp.handoff_service import HumanHandoffService

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        service = HumanHandoffService(base_url=str(client.base_url), transport=transport)

        payload = {
            "user_id": "u123",
            "conv_id": "c456",
            "latest_message": "我的订单怎么还没到",
            "intent": "query",
            "urgency": "high",
            "reason": "用户要求转人工",
            "summary": "用户连续询问物流状态。",
            "recent_messages": [
                "user: 我的订单怎么还没到",
                "assistant: 正在帮您确认订单状态",
            ],
            "user_profile": {"preferences": ["物流更新提醒"]},
            "order_snapshot": {"order_id": "ORD20250701001", "status": "运输中"},
            "knowledge_context": ["配送说明：标准配送通常 3-5 个工作日送达。"],
        }

        result = await service.handoff_handler(payload, context=None)

        self.assertEqual(result["status"], "created")
        self.assertEqual(result["queue"], "general_support")
        self.assertTrue(result["handoff_id"].startswith("HD"))

        records = api_main._mock_handoff_records
        self.assertGreaterEqual(len(records), 1)
        latest = records[-1]
        self.assertEqual(latest["user_id"], "u123")
        self.assertEqual(latest["conv_id"], "c456")
        self.assertEqual(latest["order_snapshot"]["order_id"], "ORD20250701001")
        self.assertIn("我的订单怎么还没到", latest["latest_message"])

    async def test_tool_manager_can_register_order_and_handoff_tools(self):
        from mcp.handoff_service import HumanHandoffService
        from mcp.order_lookup import OrderLookupService

        manager = MCPToolManager.__new__(MCPToolManager)
        manager._tools = {}
        manager._cache = {}

        client = TestClient(api_main.app)
        transport = httpx.ASGITransport(app=api_main.app)
        order_service = OrderLookupService(base_url=str(client.base_url), transport=transport)
        handoff_service = HumanHandoffService(base_url=str(client.base_url), transport=transport)

        manager.register(Tool(
            name="order_lookup",
            description="查询订单状态",
            handler=order_service.lookup_handler,
            schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "order_id": {"type": "string"},
                },
                "required": ["user_id", "order_id"],
            },
        ))
        manager.register(Tool(
            name="human_handoff",
            description="转人工并写入上下文",
            handler=handoff_service.handoff_handler,
            schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "conv_id": {"type": "string"},
                    "latest_message": {"type": "string"},
                },
                "required": ["user_id", "conv_id", "latest_message"],
            },
        ))

        self.assertIn("order_lookup", manager._tools)
        self.assertIn("human_handoff", manager._tools)


if __name__ == "__main__":
    unittest.main()
