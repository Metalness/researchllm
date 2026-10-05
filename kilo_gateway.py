"""
Minimal client for the Kilo AI Gateway (OpenAI-compatible).

Setup:
    pip install requests python-dotenv

Single key:
    KILO_API_KEY=your_key

Multiple keys:
    KILO_API_KEYS=key1,key2,key3

Model:
    KILO_MODEL=<exact model id>

Commands:
    python kilo_gateway.py models [filter]
    python kilo_gateway.py test

Use from other scripts:
    from kilo_gateway import chat, chat_json

    text = chat("Say hi")

    text = chat(
        "Say hi",
        api_key="specific-key"
    )

    data = chat_json(
        system_prompt,
        user_text,
        api_key="specific-key"
    )
"""

import json
import os
import re
import sys
import time

import requests
import dotenv

dotenv.load_dotenv()

BASE = "https://api.kilo.ai/api/gateway"

KEY = os.getenv("KILO_API_KEY")
MODEL = os.getenv("KILO_MODEL")


# ============================================================
# API KEY HELPERS
# ============================================================

def get_api_keys():
    """
    Read multiple keys from:

        KILO_API_KEYS=key1,key2,key3

    Falls back to:

        KILO_API_KEY=key1
    """

    multi = os.getenv(
        "KILO_API_KEYS",
        ""
    ).strip()

    if multi:

        keys = [
            key.strip()
            for key in multi.split(",")
            if key.strip()
        ]

    elif KEY:

        keys = [KEY]

    else:

        keys = []

    # Remove duplicates while preserving order
    return list(dict.fromkeys(keys))


def _headers(api_key=None):
    """
    Build request headers.

    Explicit api_key takes priority.
    """

    key = api_key or KEY

    if not key:
        raise RuntimeError(
            "Set KILO_API_KEY or KILO_API_KEYS first."
        )

    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


# ============================================================
# MODEL LIST
# ============================================================

def list_models(name_filter=None, api_key=None):

    r = requests.get(
        f"{BASE}/models",
        headers=_headers(api_key),
        timeout=30,
    )

    r.raise_for_status()

    ids = [
        m["id"]
        for m in r.json().get("data", [])
    ]

    if name_filter:

        ids = [
            i
            for i in ids
            if name_filter.lower() in i.lower()
        ]

    return sorted(ids)


# ============================================================
# CHAT
# ============================================================

def chat(
    user,
    system=None,
    model=None,
    temperature=0.2,
    max_tokens=1000,
    api_key=None,
):
    """
    Send one chat completion.

    api_key:
        Optional explicit key.

    If omitted, uses KILO_API_KEY.
    """

    model = model or MODEL

    if not model:

        raise RuntimeError(
            "Set KILO_MODEL "
            "(see: python kilo_gateway.py models nemotron)."
        )

    messages = []

    if system:

        messages.append({
            "role": "system",
            "content": system,
        })

    messages.append({
        "role": "user",
        "content": user,
    })

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last = ""

    for attempt in range(5):

        try:

            r = requests.post(
                f"{BASE}/chat/completions",
                headers=_headers(api_key),
                json=body,
                timeout=180,
            )

        except requests.RequestException as e:

            last = f"network error: {e}"

            print(
                f"  [gateway] "
                f"attempt {attempt + 1}/5 "
                f"{last}",
                flush=True,
            )

            time.sleep(
                2 ** attempt
            )

            continue

        try:

            data = r.json()

        except ValueError:

            data = None

        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        if (
            r.status_code == 200
            and isinstance(data, dict)
            and data.get("choices")
        ):

            msg = (
                data["choices"][0]
                .get("message")
                or {}
            )

            content = msg.get("content")

            if content:

                return content

            last = (
                "empty content, "
                f"finish_reason="
                f"{data['choices'][0].get('finish_reason')}"
            )

            print(
                f"  [gateway] "
                f"attempt {attempt + 1}/5 "
                f"{last}",
                flush=True,
            )

            # Reasoning models may consume all tokens.
            body["max_tokens"] = min(
                body["max_tokens"] * 2,
                8000,
            )

            continue

        # ----------------------------------------------------
        # Error response
        # ----------------------------------------------------

        last = (
            f"HTTP {r.status_code}: "
            f"{r.text[:500]}"
        )

        print(
            f"  [gateway] "
            f"attempt {attempt + 1}/5 "
            f"{last}",
            flush=True,
        )

        # Don't retry permanent request/auth/model errors.
        if r.status_code in (
            400,
            401,
            403,
            404,
        ):
            break

        time.sleep(
            2 ** attempt
        )

    raise RuntimeError(
        f"Gateway failed after retries. "
        f"Last: {last}"
    )


# ============================================================
# JSON CHAT
# ============================================================

def chat_json(
    system,
    user,
    **kw
):
    """
    Ask for JSON and parse it.

    Tolerates:
        ```json
        {...}
        ```

    and stray text around the JSON object.

    Returns:
        dict
        or None if parsing fails.
    """

    text = chat(
        user,
        system=system,
        **kw,
    )

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
    ).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    try:

        return json.loads(
            text[start:end + 1]
        )

    except json.JSONDecodeError:

        return None


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    cmd = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "test"
    )

    if cmd == "models":

        flt = (
            sys.argv[2]
            if len(sys.argv) > 2
            else None
        )

        for mid in list_models(flt):
            print(mid)

    else:

        print(
            chat(
                "Reply with exactly: gateway ok"
            )
        )