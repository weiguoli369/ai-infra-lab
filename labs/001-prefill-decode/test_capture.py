"""Protocol checks with simulated responses, never model performance results."""

import io
import json
import unittest
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from capture import capture, sse_events


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}


def event(delta=None, **choice):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, **choice}]}, ensure_ascii=False) + "\n\n"


class CaptureTests(unittest.TestCase):
    def run_stream(self, body):
        return capture("http://example.invalid/v1/chat/completions", "mock", "test",
                       opener=lambda *a, **kw: Response(body.encode("utf-8")))

    def test_role_only_chunk_does_not_start_content_clock(self):
        body = event({"role": "assistant"}) + event({"content": "你好"})
        body += event(finish_reason="stop") + "data: [DONE]\n\n"
        with patch("capture.time.perf_counter", side_effect=[10, 11, 12, 13, 14, 15]):
            result = self.run_stream(body)
        self.assertTrue(result["success"])
        self.assertEqual(result["first_text_s"], 2)
        self.assertEqual(result["elapsed_s"], 5)
        self.assertEqual(result["text"], "你好")

    def test_reasoning_and_answer_have_separate_timestamps(self):
        body = event({"reasoning_content": "想一想"}) + event({"content": "答案"})
        body += 'data: {"choices": [], "usage": {"completion_tokens": 7}}\n\n'
        result = self.run_stream(body + "data: [DONE]\n\n")
        self.assertTrue(result["success"])
        self.assertLessEqual(result["first_reasoning_s"], result["first_text_s"])
        self.assertEqual(result["usage"]["completion_tokens"], 7)
        self.assertEqual(result["reasoning"], "想一想")

    def test_sse_comments_multiline_and_final_event(self):
        stream = Response(b': keepalive\r\nevent: message\r\ndata: {\r\ndata: "choices": []}\r\n\r\ndata:[DONE]')
        self.assertEqual(list(sse_events(stream)), ['{\n"choices": []}', '[DONE]'])

    def test_truncation_preserves_partial_content(self):
        result = self.run_stream(event({"content": "partial"}))
        self.assertFalse(result["success"])
        self.assertEqual(result["text"], "partial")
        self.assertFalse(result["done_received"])

    def test_malformed_event_is_preserved(self):
        result = self.run_stream("data: invalid json\n\n")
        self.assertFalse(result["success"])
        self.assertEqual(result["events"][0]["data"], "invalid json")

    def test_network_disconnect_preserves_observed_content(self):
        class BrokenResponse(Response):
            def __iter__(self):
                yield from event({"content": "partial"}).encode("utf-8").splitlines(keepends=True)
                raise IncompleteRead(b"unfinished")
        result = capture("http://example.invalid", "mock", "test",
                         opener=lambda *a, **kw: BrokenResponse())
        self.assertFalse(result["success"])
        self.assertEqual(result["text"], "partial")
        self.assertEqual(result["error"]["type"], "IncompleteRead")

    def test_empty_or_server_error_is_failure(self):
        for body in ('data: [DONE]\n\n', 'data: {"error":{"message":"failed"}}\n\n'):
            with self.subTest(body=body):
                self.assertFalse(self.run_stream(body)["success"])

    def test_http_error_and_no_automatic_retry(self):
        calls = []
        def opener(req, **kwargs):
            calls.append(req)
            raise HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO())
        result = capture("http://example.invalid", "mock", "test", api_key="test-secret", opener=opener)
        self.assertEqual(result["http_status"], 401)
        self.assertFalse(result["success"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("test-secret", json.dumps(result))

    def test_connection_error_does_not_leak_exception_credentials(self):
        def opener(*args, **kwargs):
            raise URLError("sensitive-test-secret")
        result = capture("http://example.invalid", "mock", "test", opener=opener)
        self.assertFalse(result["success"])
        self.assertNotIn("sensitive-test-secret", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
