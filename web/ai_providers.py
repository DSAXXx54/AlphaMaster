"""AI provider resolution for training analysis.

Supports:
  - deepseek: fixed model deepseek-v4-flash @ https://api.deepseek.com (user API key)
  - openclaw: local QClaw gateway Agent (token from ~/.qclaw/openclaw.json)
  - openclaw_wb: WorkBuddy / copilot.tencent.com (local session token)

Does NOT support openclaw_cs (Cursor SDK).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

_QCLAW_CONFIG_CANDIDATES = (
    Path.home() / ".qclaw" / "openclaw.json",
    Path("~/.qclaw/openclaw.json").expanduser(),
)
_OPENCLAW_MODEL = "openclaw"

_WORKBUDDY_MODEL = "openclaw_wb"
_WORKBUDDY_API_MODEL = "auto"
_WORKBUDDY_DEFAULT_ENDPOINT = "https://copilot.tencent.com"
_WORKBUDDY_API_PATH = "/v2"
_WORKBUDDY_CONFIG_DIR = Path(
    os.environ.get("WORKBUDDY_CONFIG_DIR", "") or (Path.home() / ".workbuddy")
)
_WORKBUDDY_TOKEN_FILE = _WORKBUDDY_CONFIG_DIR / ".wb_token"

PROVIDERS = ("deepseek", "openclaw", "openclaw_wb")


@dataclass
class ResolvedProvider:
    provider: str
    model: str
    base_url: str
    api_key: str
    label: str
    needs_user_key: bool = False


def detect_qclaw() -> bool:
    info = _qclaw_gateway_info()
    if info is None:
        return False
    config_path = _find_qclaw_config()
    if config_path is None:
        return False
    data = _read_json(config_path)
    if not data:
        return False
    chat = (
        data.get("gateway", {})
        .get("http", {})
        .get("endpoints", {})
        .get("chatCompletions", {})
    )
    return bool(chat.get("enabled", False)) and bool(info[2])


def detect_workbuddy() -> bool:
    if os.environ.get("CLIENT_INFO_PRODUCT_NAME", "") == "WorkBuddy":
        return True
    if os.environ.get("WORKBUDDY_CONFIG_DIR"):
        return True
    if _workbuddy_auth_session_candidates():
        return True
    return _WORKBUDDY_CONFIG_DIR.exists()


def provider_status() -> dict[str, Any]:
    qclaw_ok = detect_qclaw()
    wb_ok = bool(_workbuddy_token())
    return {
        "providers": [
            {
                "id": "deepseek",
                "label": "DeepSeek (deepseek-v4-flash)",
                "available": True,
                "needs_user_key": True,
                "hint": "固定模型 deepseek-v4-flash · https://api.deepseek.com",
            },
            {
                "id": "openclaw",
                "label": "openclaw (QClaw)",
                "available": qclaw_ok,
                "needs_user_key": False,
                "hint": (
                    "已检测到本地 QClaw Gateway"
                    if qclaw_ok
                    else "未检测到 QClaw（需 ~/.qclaw/openclaw.json 且 chatCompletions 已启用）"
                ),
            },
            {
                "id": "openclaw_wb",
                "label": "openclaw_wb (WorkBuddy)",
                "available": wb_ok,
                "needs_user_key": False,
                "hint": (
                    "已读取 WorkBuddy 登录 token"
                    if wb_ok
                    else "未检测到 WorkBuddy token（请先登录 WorkBuddy 或设置 WORKBUDDY_API_TOKEN）"
                ),
            },
        ]
    }


def resolve_provider(provider: str, api_key: str | None = None) -> ResolvedProvider:
    pid = (provider or "deepseek").strip().lower()
    key = (api_key or "").strip()

    # API Key 里直接填 openclaw / openclaw_wb 时，自动切换通道
    key_lower = key.lower()
    if key_lower in ("openclaw",) or key_lower.startswith("openclaw/"):
        pid = "openclaw"
        key = ""
    elif key_lower in ("openclaw_wb",) or key_lower.startswith("openclaw_wb/"):
        pid = "openclaw_wb"
        key = ""

    if pid not in PROVIDERS:
        raise ValueError(f"不支持的 AI 通道: {provider}（可选: {', '.join(PROVIDERS)}）")

    if pid == "deepseek":
        if not key:
            raise ValueError("请填写 DeepSeek API Key（或在 Key 中输入 openclaw / openclaw_wb）")
        return ResolvedProvider(
            provider="deepseek",
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key=key,
            label="DeepSeek",
            needs_user_key=True,
        )

    if pid == "openclaw":
        if not detect_qclaw():
            raise ValueError(
                "未检测到本地 QClaw Gateway。请确认 QClaw 正在运行，"
                "且 ~/.qclaw/openclaw.json 中 chatCompletions 已启用、token 已配置。"
            )
        host, port, token = _qclaw_gateway_info()  # type: ignore[misc]
        base = f"http://{host}:{port}/v1"
        model = _pick_openclaw_model(base, token)
        return ResolvedProvider(
            provider="openclaw",
            model=model,
            base_url=base,
            api_key=token,
            label="openclaw (QClaw)",
            needs_user_key=False,
        )

    # openclaw_wb
    token = _workbuddy_token()
    if not token:
        raise ValueError(
            "未检测到 WorkBuddy token。请先登录 WorkBuddy，"
            "或设置环境变量 WORKBUDDY_API_TOKEN / 写入 ~/.workbuddy/.wb_token。"
        )
    endpoint = _workbuddy_endpoint()
    base = f"{endpoint.rstrip('/')}{_WORKBUDDY_API_PATH}"
    return ResolvedProvider(
        provider="openclaw_wb",
        model=_WORKBUDDY_API_MODEL,
        base_url=base,
        api_key=token,
        label="openclaw_wb (WorkBuddy)",
        needs_user_key=False,
    )


def chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> str:
    """Call OpenAI-compatible /chat/completions and return assistant text."""
    parts: list[str] = []
    for chunk in stream_chat_completions(
        resolved, messages, max_tokens=max_tokens, timeout=timeout
    ):
        parts.append(chunk)
    content = "".join(parts).strip()
    if not content:
        raise RuntimeError("AI 返回内容为空")
    return content


def stream_chat_completions(
    resolved: ResolvedProvider,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 4096,
    timeout: float = 180.0,
):
    """Yield text deltas from OpenAI-compatible streaming chat completions."""
    import urllib.error
    import urllib.request

    url = resolved.base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": resolved.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if resolved.provider == "openclaw":
        payload["tool_choice"] = "none"

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "AlphaMaster-AI-Analyze",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if not text:
                    text = delta.get("reasoning_content") or ""
                if text:
                    yield text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"AI 请求失败 HTTP {exc.code}: {err_body}") from exc
    except Exception as exc:
        raise RuntimeError(f"AI 请求失败: {exc}") from exc


def _find_qclaw_config() -> Path | None:
    for path in _QCLAW_CONFIG_CANDIDATES:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _qclaw_gateway_info() -> tuple[str, int, str] | None:
    config_path = _find_qclaw_config()
    if config_path is None:
        return None
    data = _read_json(config_path)
    if not data:
        return None
    gw = data.get("gateway") or {}
    token = str((gw.get("auth") or {}).get("token") or "").strip()
    if not token:
        return None
    port = int(gw.get("port") or 51187)
    host = "127.0.0.1"
    bind = str(gw.get("bind") or "127.0.0.1")
    if bind and bind not in ("0.0.0.0", "loopback"):
        host = bind
    return host, port, token


def _pick_openclaw_model(base_url: str, token: str) -> str:
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "AlphaMaster"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ids = [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
        if _OPENCLAW_MODEL in ids:
            return _OPENCLAW_MODEL
        for mid in ids:
            if mid.startswith("openclaw"):
                return mid
    except Exception as exc:
        logger.debug("QClaw /models probe failed: %s", exc)
    return _OPENCLAW_MODEL


def _workbuddy_auth_dir() -> Path | None:
    local_app = os.environ.get("LOCALAPPDATA", "").strip()
    if not local_app:
        return None
    auth_dir = Path(local_app) / "CodeBuddyExtension" / "Data" / "Public" / "auth"
    return auth_dir if auth_dir.is_dir() else None


def _workbuddy_auth_session_candidates() -> list[Path]:
    auth_dir = _workbuddy_auth_dir()
    if auth_dir is None:
        return []
    names = (
        os.environ.get("WORKBUDDY_AUTH_FILE", "").strip(),
        "workbuddy-desktop.info",
        "auth.info",
    )
    out: list[Path] = []
    for name in names:
        if not name:
            continue
        path = auth_dir / name
        if path.exists() and path not in out:
            out.append(path)
    return out


def _read_workbuddy_auth_token(path: Path) -> str | None:
    data = _read_json(path)
    if not data:
        return None
    auth = data.get("auth")
    if not isinstance(auth, dict):
        return None
    token = str(auth.get("accessToken") or auth.get("access_token") or "").strip()
    if not token:
        return None
    expires_at = auth.get("expiresAt")
    if isinstance(expires_at, (int, float)) and expires_at > 0:
        import time

        if expires_at <= time.time() * 1000:
            return None
    return token


def _workbuddy_token() -> str | None:
    for path in _workbuddy_auth_session_candidates():
        token = _read_workbuddy_auth_token(path)
        if token:
            return token
    if _WORKBUDDY_TOKEN_FILE.exists():
        try:
            token = _WORKBUDDY_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token:
                return token
        except OSError:
            pass
    for env_name in ("WORKBUDDY_API_TOKEN", "CODEBUDDY_AUTH_TOKEN", "ACC_AUTH_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    return None


def _workbuddy_endpoint() -> str:
    for env_name in ("WORKBUDDY_API_ENDPOINT", "WORKBUDDY_API_URL"):
        endpoint = os.environ.get(env_name, "").strip()
        if endpoint:
            return endpoint
    acc_config = os.environ.get("ACC_PRODUCT_CONFIG_V3", "")
    if acc_config:
        try:
            config = json.loads(acc_config)
            endpoint = str(config.get("endpoint") or "").strip()
            if endpoint:
                return endpoint
        except json.JSONDecodeError:
            pass
    return _WORKBUDDY_DEFAULT_ENDPOINT
