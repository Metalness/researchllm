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

Optional:
    KILO_REASONING_EFFORT=low     (low | medium | high; parameter name unverified)

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
from pathlib import Path

import requests
import dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

# Always load .env from the same directory as this script.
ENV_FILE = Path(__file__).resolve().parent / ".env"

dotenv.load_dotenv(ENV_FILE)


BASE = "https://api.kilo.ai/api/gateway"

KEY = os.getenv("KILO_API_KEY")
MODEL = os.getenv("KILO_MODEL")


# ============================================================
# CONFIG
# ============================================================

MAX_ATTEMPTS = 5

# Maximum time allowed for one HTTP request.
REQUEST_TIMEOUT = 45

# Short delays because we rotate keys.
NETWORK_RETRY_WAIT = 2
RATE_LIMIT_WAIT = 2
SERVER_ERROR_WAIT = 3


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

    # Remove duplicates while preserving order.
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

def list_models(
    name_filter=None,
    api_key=None,
):

    keys = (
        [api_key]
        if api_key
        else get_api_keys()
    )

    if not keys:
        raise RuntimeError(
            "Set KILO_API_KEY or KILO_API_KEYS first."
        )

    last = ""

    for attempt, key in enumerate(keys):

        try:

            r = requests.get(
                f"{BASE}/models",
                headers=_headers(key),
                timeout=15,
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
                    if name_filter.lower()
                    in i.lower()
                ]

            return sorted(ids)

        except requests.RequestException as e:

            last = str(e)

            print(
                f"  [gateway] models "
                f"key {attempt + 1}/{len(keys)} "
                f"failed: {e}",
                flush=True,
            )

            if attempt < len(keys) - 1:
                time.sleep(1)

    raise RuntimeError(
        f"Could not retrieve models. Last: {last}"
    )


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

    If api_key is explicitly supplied:
        only that key is used.

    Otherwise:
        rotate through KILO_API_KEYS / KILO_API_KEY
        across retries.

    Example:

        KILO_API_KEYS=key1,key2,key3

    Retry sequence:

        attempt 1 -> key1
        attempt 2 -> key2
        attempt 3 -> key3
        attempt 4 -> key1
        attempt 5 -> key2
    """

    model = model or MODEL

    if not model:

        raise RuntimeError(
            "Set KILO_MODEL "
            "(see: python kilo_gateway.py models nemotron)."
        )

    # --------------------------------------------------------
    # Determine keys
    # --------------------------------------------------------

    if api_key:

        keys = [api_key]

    else:

        keys = get_api_keys()

    if not keys:

        raise RuntimeError(
            "Set KILO_API_KEY or KILO_API_KEYS first."
        )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Optional reasoning
    # --------------------------------------------------------

    effort = os.getenv(
        "KILO_REASONING_EFFORT"
    )

    if effort:

        body["reasoning"] = {
            "effort": effort,
        }

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    last = ""

    for attempt in range(MAX_ATTEMPTS):

        # Rotate through available keys.
        key_index = attempt % len(keys)

        current_key = keys[key_index]

        # Never print the actual API key.
        key_label = (
            f"key {key_index + 1}/{len(keys)}"
        )

        print(
            f"  [gateway] "
            f"attempt {attempt + 1}/{MAX_ATTEMPTS} "
            f"using {key_label}",
            flush=True,
        )

        # ----------------------------------------------------
        # HTTP REQUEST
        # ----------------------------------------------------

        try:

            r = requests.post(
                f"{BASE}/chat/completions",
                headers=_headers(current_key),
                json=body,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as e:

            last = (
                f"network error: {e}"
            )

            print(
                f"  [gateway] "
                f"{key_label} "
                f"{last}",
                flush=True,
            )

            if attempt < MAX_ATTEMPTS - 1:

                print(
                    f"  [gateway] "
                    f"switching key in "
                    f"{NETWORK_RETRY_WAIT}s...",
                    flush=True,
                )

                time.sleep(
                    NETWORK_RETRY_WAIT
                )

            continue

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        try:

            data = r.json()

        except ValueError:

            data = None

        # ----------------------------------------------------
        # SUCCESS
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

            # Empty content.
            last = (
                "empty content, "
                f"finish_reason="
                f"{data['choices'][0].get('finish_reason')}"
            )

            print(
                f"  [gateway] "
                f"{key_label} "
                f"{last}",
                flush=True,
            )

            # Reasoning models may have consumed the
            # entire token budget.
            body["max_tokens"] = min(
                body["max_tokens"] * 2,
                32000,
            )

            if attempt < MAX_ATTEMPTS - 1:

                time.sleep(1)

            continue

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        last = (
            f"HTTP {r.status_code}: "
            f"{r.text[:500]}"
        )

        print(
            f"  [gateway] "
            f"{key_label} "
            f"{last}",
            flush=True,
        )

        # ----------------------------------------------------
        # AUTH / REQUEST ERRORS
        # ----------------------------------------------------

        if r.status_code in (
            401,
            403,
        ):

            # These may be key-specific.
            # Try the next key.
            if (
                len(keys) > 1
                and attempt < MAX_ATTEMPTS - 1
            ):

                print(
                    f"  [gateway] "
                    f"{key_label} rejected; "
                    f"switching key.",
                    flush=True,
                )

                time.sleep(1)

                continue

            break

        # 400 / 404 are generally request/model errors.
        if r.status_code in (
            400,
            404,
        ):

            break

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        rate_limited = (
            r.status_code == 429
            or '"code":429'
            in r.text[:400].replace(" ", "")
        )

        if rate_limited:

            print(
                f"  [gateway] "
                f"{key_label} rate limited; "
                f"switching key.",
                flush=True,
            )

            if attempt < MAX_ATTEMPTS - 1:

                time.sleep(
                    RATE_LIMIT_WAIT
                )

            continue

        # ----------------------------------------------------
        # OTHER SERVER ERRORS
        # ----------------------------------------------------

        if attempt < MAX_ATTEMPTS - 1:

            print(
                f"  [gateway] "
                f"server error; "
                f"switching key in "
                f"{SERVER_ERROR_WAIT}s...",
                flush=True,
            )

            time.sleep(
                SERVER_ERROR_WAIT
            )

    # --------------------------------------------------------
    # ALL RETRIES FAILED
    # --------------------------------------------------------

    raise RuntimeError(
        f"Gateway failed after "
        f"{MAX_ATTEMPTS} attempts. "
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