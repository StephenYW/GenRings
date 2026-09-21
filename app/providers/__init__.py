from app.providers.base import ImageProvider
from app.providers.mock import MockProvider


def get_provider() -> ImageProvider:
    from app import config

    if config.IMAGE_PROVIDER == "openai":
        from app.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    return MockProvider()


__all__ = ["ImageProvider", "MockProvider", "get_provider"]
