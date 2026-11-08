"""The agent brain: Claude (Anthropic) by default, Gemini as the alternative.
Follows instructions literally, no creativity."""
import os

import httpx

SYSTEM_PROMPT = """You are a simple voice agent on a phone call. You do exactly what the \
caller asks, one step at a time. You are not creative: you never invent goals, offers, or \
opinions, and you never take an action nobody asked for. You keep track of the details of \
the current task (names, addresses, amounts, times) and follow the caller's instructions. \
Speak in the caller's language, often Yiddish. Before any action that spends money or \
sends a message to someone, say exactly what you are about to do and wait for the caller \
to confirm. Keep spoken answers short: one or two sentences."""

_PROPS = {
    "send_email": ({"to": {"type": "string"}, "subject": {"type": "string"},
                    "body": {"type": "string"}}, ["to", "subject", "body"]),
    "read_recent_emails": ({"count": {"type": "integer"}}, []),
    "web_lookup": ({"url": {"type": "string"}}, ["url"]),
    "web_action": ({"site": {"type": "string"}, "action": {"type": "string"},
                    "details": {"type": "string"}}, ["site", "action", "details"]),
}

TOOLS_ANTHROPIC = [
    {"name": name, "description": desc, "input_schema": {"type": "object",
     "properties": props, "required": req}}
    for name, desc, (props, req) in [
        ("send_email", "Send an email from the agent's configured mailbox.", _PROPS["send_email"]),
        ("read_recent_emails", "Read the newest emails in the configured mailbox.", _PROPS["read_recent_emails"]),
        ("web_lookup", "Fetch a web page and return its readable text.", _PROPS["web_lookup"]),
        ("web_action", "Act on a website: search a site, place an order, or use an account. Wire real site flows in tools.py.", _PROPS["web_action"]),
    ]
]

TOOLS_GEMINI = [
    {"functionDeclarations": [
        {"name": name, "description": desc,
         "parameters": {"type": "OBJECT", "properties": {k: {"type": v.upper() if v != "integer" else "NUMBER" for k, v in props.items()}, "required": req}}
        }
        for name, desc, (props, req) in [
            ("send_email", "Send an email from the agent's configured mailbox.", _PROPS["send_email"]),
            ("read_recent_emails", "Read the newest emails in the configured mailbox.", _PROPS["read_recent_emails"]),
            ("web_lookup", "Fetch a web page and return its readable text.", _PROPS["web_lookup"]),
            ("web_action", "Act on a website: search a site, place an order, or use an account. Wire real site flows in tools.py.", _PROPS["web_action"]),
        ]
    ]}
]


class Agent:
    """Claude when ANTHROPIC_API_KEY is set, else Gemini with GOOGLE_API_KEY."""

    def __init__(self, model=None, api_key=None):
        self.anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.gemini_key = os.environ.get("GOOGLE_API_KEY")
        if self.anthropic_key:
            self.provider = "anthropic"
            self.model = model or os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
        elif self.gemini_key:
            self.provider = "gemini"
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            raise RuntimeError("Set ANTHROPIC_API_KEY (or GOOGLE_API_KEY) in .env")
        self.history = []

    def run(self, user_text: str, max_steps: int = 6) -> str:
        if self.provider == "anthropic":
            return self._run_anthropic(user_text, max_steps)
        return self._run_gemini(user_text, max_steps)

    def _run_anthropic(self, user_text: str, max_steps: int) -> str:
        messages = self.history + [{"role": "user", "content": user_text}]
        for _ in range(max_steps):
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.anthropic_key,
                         "anthropic-version": "2023-06-01"},
                json={"model": self.model, "max_tokens": 300,
                      "system": SYSTEM_PROMPT, "messages": messages,
                      "tools": TOOLS_ANTHROPIC},
                timeout=60)
            r.raise_for_status()
            data = r.json()
            blocks = data.get("content", [])
            messages.append({"role": "assistant", "content": blocks})
            calls = [b for b in blocks if b.get("type") == "tool_use"]
            if not calls:
                reply = " ".join(b.get("text", "") for b in blocks).strip()
                self.history = messages[-24:]
                return reply
            for call in calls:
                out = dispatch(call["name"], call.get("input") or {})
                messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": out}]})
        return "(stopped: too many tool steps)"

    def _run_gemini(self, user_text: str, max_steps: int) -> str:
        contents = self.history + [{"role": "user", "parts": [{"text": user_text}]}]
        for _ in range(max_steps):
            body = {
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": contents,
                "tools": TOOLS_GEMINI,
            }
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{self.model}:generateContent")
            r = httpx.post(url, params={"key": self.gemini_key}, json=body, timeout=60)
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            contents.append({"role": "model", "parts": parts})
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                reply = " ".join(p.get("text", "") for p in parts).strip()
                self.history = contents[-24:]
                return reply
            for call in calls:
                out = dispatch(call["name"], call.get("args", {}))
                contents.append({"role": "user", "parts": [
                    {"functionResponse": {"name": call["name"],
                                          "response": {"result": out}}}]})
        return "(stopped: too many tool steps)"


def dispatch(name: str, args: dict) -> str:
    """Run one tool call; never let a tool error crash the call."""
    import tools
    fn = getattr(tools, name, None)
    if fn is None:
        return f"unknown tool: {name}"
    try:
        return str(fn(**args))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
