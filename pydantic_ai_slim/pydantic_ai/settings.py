from __future__ import annotations

from typing import Union, Any, Annotated
from httpx import Timeout
from typing_extensions import TypedDict
import pydantic_core
from pydantic import GetCoreSchemaHandler


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


class _TimeoutPydanticAnnotation:
    """Annotation to make httpx.Timeout work with Pydantic."""
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> pydantic_core.CoreSchema:
        """Generate Pydantic core schema for httpx.Timeout.
        
        This allows Pydantic to handle httpx.Timeout objects in model fields
        by providing custom serialization and deserialization logic.
        """
        from pydantic_core import core_schema
        
        def validate_timeout(value: Any) -> Union[float, Timeout]:
            """Validate and convert input to float or Timeout."""
            if isinstance(value, Timeout):
                return value
            elif isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, dict):
                # Convert dict to Timeout - use deserialize_timeout which already handles this
                return deserialize_timeout(value)
            else:
                raise ValueError(f"Expected float, Timeout, or dict, got {type(value)}")
        
        # Define the schema for the serialized form
        timeout_dict_schema = core_schema.dict_schema(
            keys_schema=core_schema.str_schema(),
            values_schema=core_schema.float_schema(),
        )
        
        # Union of float or dict for the serialized form
        serialized_schema = core_schema.union_schema([
            core_schema.float_schema(),
            timeout_dict_schema,
        ])
        
        # Use a custom validator that can handle both Python and JSON inputs
        return core_schema.with_info_after_validator_function(
            function=lambda v, _: validate_timeout(v),
            schema=core_schema.union_schema([
                core_schema.float_schema(),
                core_schema.dict_schema(),
                core_schema.is_instance_schema(Timeout),
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_timeout,
                info_arg=False,
                return_schema=serialized_schema,
                when_used='json'
            )
        )


# Create a type alias that includes the Pydantic annotation
TimeoutType = Annotated[Union[float, Timeout], _TimeoutPydanticAnnotation]


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

    timeout: TimeoutType
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
