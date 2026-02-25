import os
import time


PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai":     "https://api.openai.com/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama":     "http://localhost:11434/v1",
}

DEFAULT_MODELS = {
    "openrouter": "arcee-ai/trinity-large-preview:free",
    "openai":     "gpt-4o-mini",
    "gemini":     "gemini-2.0-flash",
    "ollama":     "llama3",
}

ENV_KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "ollama":     None,
}


def ask(prompt, model=None, api_key=None, base_url=None, provider=None, retries=3, schema=None):
    try:
        from openai import OpenAI, RateLimitError

        provider = _detect_provider(provider, model, base_url)
        model = model or DEFAULT_MODELS[provider]
        base_url = base_url or PROVIDERS[provider]
        api_key = _resolve_key(api_key, provider)
        client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

        if schema:
            prompt = prompt + f"\n\nRespond ONLY with a valid JSON object matching this schema: {schema.model_json_schema()}"

        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
        if schema:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            try:
                response = client.chat.completions.create(**kwargs)
                text = response.choices[0].message.content
                if schema:
                    import json
                    return schema.model_validate(json.loads(text))
                return text
            except RateLimitError:
                if attempt < retries - 1:
                    wait = 5 * (attempt + 1)
                    print(f"[cofone] rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    except ImportError:
        raise ImportError("pip install openai")


def stream(prompt, model=None, api_key=None, base_url=None, provider=None):
    try:
        from openai import OpenAI

        provider = _detect_provider(provider, model, base_url)
        model = model or DEFAULT_MODELS[provider]
        base_url = base_url or PROVIDERS[provider]
        api_key = _resolve_key(api_key, provider)
        client = OpenAI(api_key=api_key or "ollama", base_url=base_url)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except ImportError:
        raise ImportError("pip install openai")


def _detect_provider(provider, model, base_url):
    if provider:
        return provider
    if base_url:
        for name, url in PROVIDERS.items():
            if url in base_url:
                return name
        return "openrouter"
    if model:
        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            return "openai"
        if model.startswith("gemini"):
            return "gemini"
        if "/" in model:
            return "openrouter"
    return "openrouter"


def _resolve_key(api_key, provider):
    if api_key:
        return api_key
    env = ENV_KEYS.get(provider)
    if env:
        key = os.environ.get(env)
        if not key:
            raise ValueError(f"[cofone] API key not found for provider '{provider}'.\nSet {env} in your .env file.")
        return key
    return "ollama"