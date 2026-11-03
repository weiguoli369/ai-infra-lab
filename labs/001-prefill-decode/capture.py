#!/usr/bin/env python3
"""Capture one streaming chat request with Python's standard library."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib import error, parse, request


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Avoid forwarding an Authorization header to an unexpected destination.
        return None


def sse_events(response):
    """Yield complete SSE data fields; preserve multiline events and UTF-8."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def capture(url, model, prompt, max_tokens=64, timeout=120.0,
            include_usage=False, api_key=None, opener=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    result = {
        "run_id": uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "client": {"python": platform.python_version(),
                   "system": platform.system(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "endpoint": url,
        "request": payload,
        "timeout_per_socket_operation_s": timeout,
        "server_environment": "unknown; record separately",
        "http_status": None,
        "success": False,
        "done_received": False,
        "first_text_s": None,
        "first_reasoning_s": None,
        "first_any_content_s": None,
        "elapsed_s": None,
        "usage": None,
        "finish_reason": None,
        "text": "",
        "reasoning": "",
        "events": [],
        "error": None,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    open_request = opener or request.build_opener(NoRedirect()).open
    start = time.perf_counter()
    try:
        with open_request(req, timeout=timeout) as response:
            result["http_status"] = response.status
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
            if media_type != "text/event-stream":
                raise ValueError("Expected text/event-stream response")
            for event in sse_events(response):
                elapsed = time.perf_counter() - start
                result["events"].append({"received_s": elapsed, "data": event})
                if event.strip() == "[DONE]":
                    result["done_received"] = True
                    break
                chunk = json.loads(event)
                if not isinstance(chunk, dict) or "error" in chunk:
                    raise ValueError("Invalid chunk or server stream error; inspect saved events")
                if chunk.get("usage") is not None:
                    result["usage"] = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("index", 0) != 0:
                        continue
                    if choice.get("finish_reason") is not None:
                        result["finish_reason"] = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    for field, output, timing in (
                        ("content", "text", "first_text_s"),
                        ("reasoning_content", "reasoning", "first_reasoning_s"),
                        ("reasoning", "reasoning", "first_reasoning_s"),
                    ):
                        content = delta.get(field)
                        if isinstance(content, str) and content:
                            result[output] += content
                            if result[timing] is None:
                                result[timing] = elapsed
                            if result["first_any_content_s"] is None:
                                result["first_any_content_s"] = elapsed
        if not result["done_received"]:
            raise ValueError("Stream ended without [DONE]; possibly interrupted")
        if result["first_any_content_s"] is None:
            raise ValueError("Stream completed without text or reasoning content")
        result["success"] = True
    except error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = {"type": "HTTPError", "message": "HTTP status %s" % exc.code}
        exc.close()
    except (error.URLError, HTTPException, OSError, ValueError, TypeError, AttributeError) as exc:
        # Exception strings can contain URLs or credentials; retain type, not secrets.
        message = str(exc) if type(exc) is ValueError else "Request or stream failed; check service and network"
        result["error"] = {"type": type(exc).__name__, "message": message}
    finally:
        result["elapsed_s"] = time.perf_counter() - start
    return result


def main():
    parser = argparse.ArgumentParser(description="采集一次流式请求；不进行自动重试或并发压测")
    parser.add_argument("--url", required=True, help="完整 /v1/chat/completions 地址")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请用一句话解释 GPU 的作用。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--include-usage", action="store_true", help="服务支持时请求 usage")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    url = parse.urlsplit(args.url)
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("--url 必须是完整的 http/https 地址")
    if url.username or url.password or url.query or url.fragment:
        parser.error("地址不能包含凭证、查询参数或片段；密钥放在 LAB_API_KEY 环境变量")
    if args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("max-tokens 和 timeout 必须大于零，timeout 必须是有限值")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = capture(args.url, args.model, args.prompt, args.max_tokens,
                     args.timeout, args.include_usage, os.getenv("LAB_API_KEY"))
    path = args.output_dir / (result["run_id"] + ".json")
    with path.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print("成功:", result["success"], "HTTP:", result["http_status"])
    for label, key in (("首段正文", "first_text_s"),
                       ("首段思考", "first_reasoning_s"),
                       ("首段任意内容", "first_any_content_s"),
                       ("完整请求", "elapsed_s")):
        value = result[key]
        print(label + ":", "未收到" if value is None else "%.3f 秒" % value)
    print("finish_reason:", result["finish_reason"], "usage:", result["usage"])
    print("正文预览:", result["text"][:200])
    print("记录:", path.resolve())
    if result["error"]:
        print("错误:", result["error"])
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
