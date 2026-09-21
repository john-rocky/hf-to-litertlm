"""Deterministic tool-probe fixtures; no system message and no network activity."""


def get_weather(location: str):
    """Get the current weather for a location.

    Args:
        location: City or location whose weather is requested.
    """
    return {"location": location, "temperature_c": 18, "condition": "clear",
            "source": "controlled test fixture; not live weather"}


def multiply(a: float, b: float):
    """Multiply two numbers.

    Args:
        a: First number to multiply.
        b: Second number to multiply.
    """
    return {"a": a, "b": b, "product": float(a) * float(b)}


def web_search(query: str):
    """Search the web for information matching a query.

    Args:
        query: Search terms describing the information to find.
    """
    return {"query": query, "results": [{"title": "LiteRT-LM releases (test fixture)",
            "url": "https://github.com/google-ai-edge/LiteRT-LM/releases",
            "snippet": "Controlled tool-probe fixture; no live search was performed."}], "test_fixture": True}


tools = [get_weather, multiply, web_search]
