from .openai_provider import OpenAIProvider
from .generic_http_provider import GenericHTTPProvider
from .local_provider import LocalProvider
from .gemini_provider import GeminiProvider
from .gorq_provider import GorqProvider

def get_provider(name: str, **kwargs):
    if name == 'openai':
        return OpenAIProvider(**kwargs.get('openai', {}))
    if name == 'gemini':
        return GeminiProvider(**kwargs.get('gemini', {}))
    if name == 'gorq':
        return GorqProvider(**kwargs.get('gorq', {}))
    if name == 'generic':
        return GenericHTTPProvider(**kwargs.get('generic', {}))
    if name == 'local':
        return LocalProvider(**kwargs.get('local', {}))
    raise ValueError(f'Unknown provider: {name}')
