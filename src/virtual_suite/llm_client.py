"""Client for an explicitly configured OpenAI-compatible local endpoint.
VLLM_BASE_URL defaults to http://localhost:8000/v1; VLLM_MODEL selects the served
model. Defaults use greedy decoding (temperature 0, seed 0). Requests explicitly
disable top-p/top-k filtering for temperature 0, matching vLLM greedy decoding.
"""
from __future__ import annotations
import json, os, time, urllib.error, urllib.request

DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}   # running tally of this process, available to the caller


def settings() -> dict:
    return {"backend": "local", "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"), "model": os.environ.get("VLLM_MODEL", DEFAULT_MODEL)}


def chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0, seed: int | None = 0, retries: int = 5, timeout: int = 600, extra: dict | None = None, tools: list | None = None, tool_choice: str = "auto") -> dict:
    """Returns {"content": str, "reasoning": str|None, "tool_calls": list, "message": dict, "usage": dict, "model": str, "latency_s": float, "backend": str, "provider": None}.
    Greedy decoding by default (temperature 0, seed 0)."""
    s = settings(); body = {"model": s["model"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools: body["tools"] = tools; body["tool_choice"] = tool_choice      # native function calling: the reply carries `tool_calls` and the assistant `message` to append
    if seed is not None: body["seed"] = seed
    if extra: body.update(extra)            # e.g. {"chat_template_kwargs": {"thinking": False}}
    if body["temperature"] == 0:
        body.update(top_p=1.0, top_k=0)  # vLLM greedy decoding; no top-p/top-k filtering
    req = urllib.request.Request(f"{s['base_url']}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "User-Agent": "virtual-suite/0.1"})
    attempt = 0
    while True:
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: d = json.load(r)
            u = d.get("usage") or {}; USAGE["calls"] += 1
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"): USAGE[k] += int(u.get(k) or 0)
            m = d["choices"][0]["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning") or m.get("reasoning_content"), "tool_calls": m.get("tool_calls") or [], "message": m,
                    "usage": u, "model": d.get("model", s["model"]), "latency_s": time.time() - t0, "backend": s["backend"], "provider": None}
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError):
            attempt += 1
            if attempt >= retries: raise
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))   # a server restarting or a full queue clears within a minute


def check() -> None:
    """The model is listed, answers, and returns a native tool call (the agents need --enable-auto-tool-choice and the hermes parser)."""
    s = settings()
    with urllib.request.urlopen(f"{s['base_url']}/models", timeout=30) as r: ids = [m["id"] for m in json.load(r)["data"]]
    if s["model"] not in ids: raise SystemExit(f"model {s['model']!r} not served at {s['base_url']} (served: {ids}); set VLLM_MODEL")
    off = {"chat_template_kwargs": {"thinking": False}}
    r = chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=8, extra=off)
    print(f"reply: {r['content']!r}  ({r['usage'].get('total_tokens')} tokens, {r['latency_s']:.1f}s)")
    tool = {"type": "function", "function": {"name": "forecast", "description": "Forecast when an operating room finishes.",
            "parameters": {"type": "object", "properties": {"room": {"type": "integer"}}, "required": ["room"]}}}
    r = chat([{"role": "user", "content": "When will room 3 finish? Use the tool."}], max_tokens=64, extra=off, tools=[tool], tool_choice="required")
    if not r["tool_calls"]: raise SystemExit("no tool call returned: start vLLM with --enable-auto-tool-choice --tool-call-parser hermes (enable automatic tool choice and the hermes tool-call parser)")
    f = r["tool_calls"][0]["function"]; print(f"tool call: {f['name']}({f['arguments']})"); print(f"OK: {s['model']} at {s['base_url']}")


if __name__ == "__main__":
    check()
