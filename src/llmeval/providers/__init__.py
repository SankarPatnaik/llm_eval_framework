from .openai_provider import OpenAIProvider
from .generic_http_provider import GenericHTTPProvider
from .local_provider import LocalProvider

def get_provider(name: str, **kwargs):
    if name == 'openai':
        return OpenAIProvider(**kwargs.get('openai', {}))
    if name == 'generic':
        return GenericHTTPProvider(**kwargs.get('generic', {}))
    if name == 'local':
        return LocalProvider(**kwargs.get('local', {}))
    raise ValueError(f'Unknown provider: {name}')
