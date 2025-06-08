from __future__ import annotations

from typing import Union
from httpx import Timeout
from typing_extensions import TypedDict


def serialize_timeout(timeout: Union[float, Timeout]) -> Union[float, dict[str, float]]:
    """Convert an httpx.Timeout object to a serializable format.
    
    Args:
        timeout: Either a float or an httpx.Timeout object
        
    Returns:
        Either a float (for simple timeouts) or a dict with timeout components
    """
    if isinstance(timeout, (int, float)):
        return float(timeout)
    elif isinstance(timeout, Timeout):
        # Extract the individual timeout values
        result: dict[str, float] = {}
        if timeout.connect is not None:
            result['connect'] = timeout.connect
        if timeout.read is not None:
            result['read'] = timeout.read
        if timeout.write is not None:
            result['write'] = timeout.write
        if timeout.pool is not None:
            result['pool'] = timeout.pool
        return result if result else 5.0  # Default timeout if all are None
    else:
        raise TypeError(f"Expected float or Timeout, got {type(timeout)}")


def deserialize_timeout(value: Union[float, dict[str, float]]) -> Union[float, Timeout]:
    """Convert a serialized timeout value back to float or httpx.Timeout.
    
    Args:
        value: Either a float or a dict with timeout components
        
    Returns:
        Either a float or an httpx.Timeout object
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, dict):
        # Create Timeout with specific values
        return Timeout(
            connect=value.get('connect'),
            read=value.get('read'),
            write=value.get('write'),
            pool=value.get('pool')
        )
    else:
        raise TypeError(f"Expected float or dict, got {type(value)}")


class ModelSettings(TypedDict, total=False):
    """Settings to configure an LLM.

    Here we include only settings which apply to multiple models / model providers,
    though not all of these settings are supported by all models.
    """

    max_tokens: int
    """The maximum number of tokens to generate before stopping.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    """

    temperature: float
    """Amount of randomness injected into the response.

    Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
    maximum `temperature` for creative and generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully deterministic.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    """

    top_p: float
    """An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.

    So 0.1 means only the tokens comprising the top 10% probability mass are considered.

    You should either alter `temperature` or `top_p`, but not both.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Cohere
    * Mistral
    * Bedrock
    """

    timeout: float | Timeout
    """Override the client-level default timeout for a request, in seconds.

    Supported by:

    * Gemini
    * Anthropic
    * OpenAI
    * Groq
    * Mistral
    """

    parallel_tool_calls: bool
    """Whether to allow parallel tool calls.

    Supported by:

    * OpenAI (some models, not o1)
    * Groq
    * Anthropic
    """

    seed: int
    """The random seed to use for the model, theoretically allowing for deterministic results.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Mistral
    """

    presence_penalty: float
    """Penalize new tokens based on whether they have appeared in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    frequency_penalty: float
    """Penalize new tokens based on their existing frequency in the text so far.

    Supported by:

    * OpenAI
    * Groq
    * Cohere
    * Gemini
    * Mistral
    """

    logit_bias: dict[str, int]
    """Modify the likelihood of specified tokens appearing in the completion.

    Supported by:

    * OpenAI
    * Groq
    """

    stop_sequences: list[str]
    """Sequences that will cause the model to stop generating.

    Supported by:

    * OpenAI
    * Anthropic
    * Bedrock
    * Mistral
    * Groq
    * Cohere
    * Google
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """

    extra_body: object
    """Extra body to send to the model.

    Supported by:

    * OpenAI
    * Anthropic
    * Groq
    """


def merge_model_settings(base: ModelSettings | None, overrides: ModelSettings | None) -> ModelSettings | None:
    """Merge two sets of model settings, preferring the overrides.

    A common use case is: merge_model_settings(<agent settings>, <run settings>)
    """
    # Note: we may want merge recursively if/when we add non-primitive values
    if base and overrides:
        return base | overrides
    else:
        return base or overrides
