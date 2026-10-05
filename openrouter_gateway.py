"""
Minimal OpenRouter gateway.

Setup:
    pip install requests python-dotenv

Single key:
    OPENROUTER_API_KEY=your_key

Multiple keys:
    OPENROUTER_API_KEYS=key1,key2,key3

Default model:
    OPENROUTER_MODEL=<exact OpenRouter model id>

Per-call models:
    OPENROUTER_MODEL_CALL1=<model>
    OPENROUTER_MODEL_CALL2=<model>

Per-call reasoning:
    OPENROUTER_REASONING_CALL1=false
    OPENROUTER_REASONING_EFFORT_CALL1=low

    OPENROUTER_REASONING_CALL2=true
    OPENROUTER_REASONING_EFFORT_CALL2=medium

Optional:
    OPENROUTER_SITE_URL=https://your-site.com
    OPENROUTER_SITE_NAME=Your App

Use:
    from openrouter_gateway import chat, chat_json

    text = chat("Say hi")

    text = chat(
        "Say hi",
        model="some/model",
        reasoning=True,
        reasoning_effort="medium",
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

ENV_FILE = (
    Path(__file__).resolve().parent
    / ".env"
)

dotenv.load_dotenv(ENV_FILE)


BASE = "https://openrouter.ai/api"

KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

MODEL = os.getenv(
    "OPENROUTER_MODEL"
)


# ============================================================
# CONFIG
# ============================================================

MAX_ATTEMPTS = int(
    os.getenv(
        "OPENROUTER_MAX_ATTEMPTS",
        "5",
    )
)

REQUEST_TIMEOUT = int(
    os.getenv(
        "OPENROUTER_REQUEST_TIMEOUT",
        "45",
    )
)

NETWORK_RETRY_WAIT = float(
    os.getenv(
        "OPENROUTER_NETWORK_RETRY_WAIT",
        "2",
    )
)

RATE_LIMIT_WAIT = float(
    os.getenv(
        "OPENROUTER_RATE_LIMIT_WAIT",
        "2",
    )
)

SERVER_ERROR_WAIT = float(
    os.getenv(
        "OPENROUTER_SERVER_ERROR_WAIT",
        "3",
    )
)


# ============================================================
# API KEY HELPERS
# ============================================================

def get_api_keys():
    """
    Read:

        OPENROUTER_API_KEYS=key1,key2,key3

    Falls back to:

        OPENROUTER_API_KEY=key1
    """

    multi = os.getenv(
        "OPENROUTER_API_KEYS",
        "",
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
    return list(
        dict.fromkeys(keys)
    )


def _headers(api_key=None):
    """
    Build OpenRouter request headers.
    """

    key = api_key or KEY

    if not key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY "
            "or OPENROUTER_API_KEYS first."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv(
        "OPENROUTER_SITE_URL"
    )

    site_name = os.getenv(
        "OPENROUTER_SITE_NAME"
    )

    if site_url:
        headers["HTTP-Referer"] = site_url

    if site_name:
        headers["X-Title"] = site_name

    return headers


# ============================================================
# MODEL LIST
# ============================================================

def list_models(
    name_filter=None,
    api_key=None,
):
    """
    List OpenRouter models.
    """

    keys = (
        [api_key]
        if api_key
        else get_api_keys()
    )

    if not keys:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY "
            "or OPENROUTER_API_KEYS first."
        )

    last = ""

    for attempt, key in enumerate(keys):

        try:
            response = requests.get(
                f"{BASE}/v1/models",
                headers=_headers(key),
                timeout=15,
            )

            response.raise_for_status()

            ids = [
                model["id"]
                for model in response.json().get(
                    "data",
                    [],
                )
            ]

            if name_filter:
                ids = [
                    model_id
                    for model_id in ids
                    if name_filter.lower()
                    in model_id.lower()
                ]

            return sorted(ids)

        except requests.RequestException as e:

            last = str(e)

            print(
                f"  [openrouter] models "
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
    reasoning=None,
    reasoning_effort=None,
):
    """
    Send one OpenRouter chat completion.

    Parameters
    ----------
    user:
        User prompt.

    system:
        Optional system prompt.

    model:
        Optional model override.

    temperature:
        Sampling temperature.

    max_tokens:
        Maximum output tokens.

    api_key:
        Explicit key. If omitted, configured keys rotate.

    reasoning:
        None:
            Use normal request behavior.

        True:
            Enable reasoning.

        False:
            Disable reasoning.

    reasoning_effort:
        low / medium / high

    Important behavior:

    If the model returns finish_reason="length",
    the request is NOT retried.

    A token-limit failure is a deterministic failure of
    the current request, not a transient network/key error.
    """

    model = model or MODEL

    if not model:
        raise RuntimeError(
            "Set OPENROUTER_MODEL first."
        )

    # --------------------------------------------------------
    # Keys
    # --------------------------------------------------------

    if api_key:
        keys = [api_key]
    else:
        keys = get_api_keys()

    if not keys:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY "
            "or OPENROUTER_API_KEYS first."
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

    # --------------------------------------------------------
    # Request body
    # --------------------------------------------------------

    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    if reasoning is True:

        effort = (
            reasoning_effort
            or os.getenv(
                "OPENROUTER_REASONING_EFFORT",
                "medium",
            )
        )

        body["reasoning"] = {
            "enabled": True,
            "effort": effort,
        }

    elif reasoning is False:

        body["reasoning"] = {
            "enabled": False,
        }

    else:

        # Backwards-compatible behavior.
        #
        # If reasoning was not explicitly specified by the
        # caller, use the global environment variable.
        effort = os.getenv(
            "OPENROUTER_REASONING_EFFORT"
        )

        if effort:
            body["reasoning"] = {
                "effort": effort,
            }

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    print(
        f"  [openrouter] model={model}",
        flush=True,
    )

    if "reasoning" in body:
        print(
            f"  [openrouter] reasoning="
            f"{body['reasoning']}",
            flush=True,
        )
    else:
        print(
            "  [openrouter] reasoning=default",
            flush=True,
        )

    print(
        f"  [openrouter] max_tokens="
        f"{body['max_tokens']}",
        flush=True,
    )

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    last = ""

    for attempt in range(MAX_ATTEMPTS):

        key_index = (
            attempt % len(keys)
        )

        current_key = keys[key_index]

        key_label = (
            f"key {key_index + 1}/{len(keys)}"
        )

        print(
            f"  [openrouter] "
            f"attempt {attempt + 1}/{MAX_ATTEMPTS} "
            f"using {key_label}",
            flush=True,
        )

        start_time = time.time()

        # ----------------------------------------------------
        # HTTP REQUEST
        # ----------------------------------------------------

        try:

            response = requests.post(
                f"{BASE}/v1/chat/completions",
                headers=_headers(
                    current_key
                ),
                json=body,
                timeout=REQUEST_TIMEOUT,
            )

        except requests.RequestException as e:

            elapsed = (
                time.time() - start_time
            )

            last = (
                f"network error: {e}"
            )

            print(
                f"  [openrouter] "
                f"{key_label} "
                f"network error after "
                f"{elapsed:.1f}s: {e}",
                flush=True,
            )

            if attempt < MAX_ATTEMPTS - 1:

                print(
                    f"  [openrouter] "
                    f"retrying in "
                    f"{NETWORK_RETRY_WAIT}s...",
                    flush=True,
                )

                time.sleep(
                    NETWORK_RETRY_WAIT
                )

            continue

        elapsed = (
            time.time() - start_time
        )

        print(
            f"  [openrouter] "
            f"{key_label} "
            f"HTTP {response.status_code} "
            f"after {elapsed:.1f}s",
            flush=True,
        )

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        try:
            data = response.json()

        except ValueError:
            data = None

        # ----------------------------------------------------
        # SUCCESS RESPONSE
        # ----------------------------------------------------

        if (
            response.status_code == 200
            and isinstance(data, dict)
            and data.get("choices")
        ):

            choice = data["choices"][0]

            finish_reason = choice.get(
                "finish_reason"
            )

            message = (
                choice.get("message")
                or {}
            )

            content = message.get(
                "content"
            )

            # ------------------------------------------------
            # Normal successful content
            # ------------------------------------------------

            if content:

                print(
                    f"  [openrouter] "
                    f"success: "
                    f"{len(content)} chars, "
                    f"finish_reason="
                    f"{finish_reason}",
                    flush=True,
                )

                return content

            # ------------------------------------------------
            # TOKEN LIMIT
            # ------------------------------------------------

            if finish_reason == "length":

                last = (
                    "model reached max_tokens "
                    f"({body['max_tokens']}) "
                    "with no usable content"
                )

                print(
                    f"  [openrouter] "
                    f"{key_label} "
                    f"{last}",
                    flush=True,
                )

                # DO NOT RETRY.
                #
                # Retrying the exact same request just wastes
                # several minutes and produces the same result.
                raise RuntimeError(
                    "OpenRouter output hit "
                    f"max_tokens={body['max_tokens']} "
                    "before producing usable content. "
                    "Increase max_tokens or reduce the "
                    "requested reasoning/output."
                )

            # ------------------------------------------------
            # OTHER EMPTY RESPONSE
            # ------------------------------------------------

            last = (
                "empty content, "
                f"finish_reason={finish_reason}"
            )

            print(
                f"  [openrouter] "
                f"{key_label} "
                f"{last}",
                flush=True,
            )

            # Empty content without length may be transient.
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(1)

            continue

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if isinstance(data, dict):
            error_text = json.dumps(
                data,
                ensure_ascii=False,
            )[:1000]
        else:
            error_text = (
                response.text[:1000]
            )

        last = (
            f"HTTP {response.status_code}: "
            f"{error_text}"
        )

        print(
            f"  [openrouter] "
            f"{key_label} "
            f"{last}",
            flush=True,
        )

        # ----------------------------------------------------
        # AUTH ERRORS
        # ----------------------------------------------------

        if response.status_code in (
            401,
            403,
        ):

            if (
                len(keys) > 1
                and attempt < MAX_ATTEMPTS - 1
            ):

                print(
                    f"  [openrouter] "
                    f"{key_label} rejected; "
                    f"switching key.",
                    flush=True,
                )

                time.sleep(1)

                continue

            break

        # ----------------------------------------------------
        # BAD REQUEST / NOT FOUND
        # ----------------------------------------------------

        if response.status_code in (
            400,
            404,
        ):

            # These are generally deterministic request
            # errors. Do not waste retries.
            break

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        rate_limited = (
            response.status_code == 429
            or '"code":429'
            in response.text[:400]
            .replace(" ", "")
        )

        if rate_limited:

            print(
                f"  [openrouter] "
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
        # SERVER ERROR
        # ----------------------------------------------------

        if attempt < MAX_ATTEMPTS - 1:

            print(
                f"  [openrouter] "
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
        f"OpenRouter failed after "
        f"{MAX_ATTEMPTS} attempts. "
        f"Last: {last}"
    )


# ============================================================
# JSON CHAT
# ============================================================

def chat_json(
    system,
    user,
    **kwargs,
):
    """
    Ask for JSON and parse it.

    Accepts the same keyword arguments as chat().
    """

    text = chat(
        user,
        system=system,
        **kwargs,
    )

    text = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in response:\n"
            + text[:2000]
        )

    try:
        return json.loads(
            text[start:end + 1]
        )

    except json.JSONDecodeError as e:

        raise ValueError(
            "Could not parse JSON response:\n"
            + text[:3000]
        ) from e


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    command = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "test"
    )

    if command == "models":

        name_filter = (
            sys.argv[2]
            if len(sys.argv) > 2
            else None
        )

        for model_id in list_models(
            name_filter
        ):
            print(model_id)

    elif command == "test":

        print(
            chat(
                "Reply with exactly: gateway ok"
            )
        )

    else:

        print(
            "Usage:"
        )

        print(
            "  python "
            "openrouter_gateway.py models "
            "[filter]"
        )

        print(
            "  python "
            "openrouter_gateway.py test"
        )
