from __future__ import annotations as _annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from functools import cached_property
from typing import Annotated, Any, TypeVar, cast

import httpx
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    Citation,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UrlCitationSource,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai.builtin_tools import (
    CodeExecutionTool,
    MCPServerTool,
    MemoryTool,
    ShellTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
    InstructionPart,
    UploadedFile,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets.shell import ShellToolset
from pydantic_ai.toolsets.text_editor import TextEditorCommand, TextEditorOutput, TextEditorToolset
from pydantic_ai.usage import RequestUsage, UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, TestEnv, raise_if_exception, try_import
from ..parts_from_messages import part_types_from_messages
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncAnthropic
    from anthropic.lib.tools import BetaAbstractMemoryTool
    from anthropic.resources.beta import AsyncBeta
    from anthropic.types.beta import (
        BetaCodeExecutionResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaContentBlock,
        BetaDirectCaller,
        BetaInputJSONDelta,
        BetaMemoryTool20250818CreateCommand,
        BetaMemoryTool20250818DeleteCommand,
        BetaMemoryTool20250818InsertCommand,
        BetaMemoryTool20250818RenameCommand,
        BetaMemoryTool20250818StrReplaceCommand,
        BetaMemoryTool20250818ViewCommand,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaMessageTokensCount,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaTextDelta,
        BetaTextEditorCodeExecutionToolResultBlock,
        BetaToolUseBlock,
        BetaUsage,
        BetaWebSearchResultBlock,
        BetaWebSearchToolResultBlock,
    )
    from anthropic.types.beta.beta_container import BetaContainer
    from anthropic.types.beta.beta_container_params import BetaContainerParams
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _map_usage,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    MockAnthropicMessage = BetaMessage | Exception
    MockRawMessageStreamEvent = BetaRawMessageStreamEvent | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]

# Type variable for generic AsyncStream
T = TypeVar('T')


def test_init():
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key='foobar'))
    assert isinstance(m.client, AsyncAnthropic)
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'claude-haiku-4-5'
    assert m.system == 'anthropic'
    assert m.base_url == 'https://api.anthropic.com'


@dataclass
class MockAnthropic:
    messages_: MockAnthropicMessage | Sequence[MockAnthropicMessage] | None = None
    stream: Sequence[MockRawMessageStreamEvent] | Sequence[Sequence[MockRawMessageStreamEvent]] | None = None
    index = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    base_url: str = 'https://api.anthropic.com'

    @cached_property
    def beta(self) -> AsyncBeta:
        return cast(AsyncBeta, self)

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.messages_create, 'count_tokens': self.messages_count_tokens})

    @classmethod
    def create_mock(cls, messages_: MockAnthropicMessage | Sequence[MockAnthropicMessage]) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(messages_=messages_))

    @classmethod
    def create_stream_mock(
        cls, stream: Sequence[MockRawMessageStreamEvent] | Sequence[Sequence[MockRawMessageStreamEvent]]
    ) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(stream=stream))

    async def messages_create(
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> BetaMessage | MockAsyncStream[MockRawMessageStreamEvent]:
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        if stream:
            assert self.stream is not None, 'you can only use `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockRawMessageStreamEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockRawMessageStreamEvent], self.stream)))
        else:
            assert self.messages_ is not None, '`messages` must be provided'
            if isinstance(self.messages_, Sequence):
                raise_if_exception(self.messages_[self.index])
                response = cast(BetaMessage, self.messages_[self.index])
            else:
                raise_if_exception(self.messages_)
                response = cast(BetaMessage, self.messages_)
        self.index += 1
        return response

    async def messages_count_tokens(self, *_args: Any, **kwargs: Any) -> BetaMessageTokensCount:
        # check if we are configured to raise an exception
        if self.messages_ is not None:
            raise_if_exception(self.messages_ if not isinstance(self.messages_, Sequence) else self.messages_[0])

        # record the kwargs used
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        return BetaMessageTokensCount(input_tokens=10)


def completion_message(content: list[BetaContentBlock], usage: BetaUsage) -> BetaMessage:
    return BetaMessage(
        id='123',
        content=content,
        model='claude-3-5-haiku-123',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=usage,
    )


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=5,
            output_tokens=10,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=5,
            output_tokens=10,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_pause_turn_continues_run(allow_model_requests: None):
    c1 = completion_message([BetaTextBlock(text='paused', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    c1.stop_reason = 'pause_turn'
    c2 = completion_message([BetaTextBlock(text='final', type='text')], BetaUsage(input_tokens=10, output_tokens=5))

    mock_client = MockAnthropic.create_mock([c1, c2])
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    result = await agent.run('test prompt')

    assert result.output == 'final'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test prompt', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final')],
                usage=RequestUsage(input_tokens=10, output_tokens=5, details={'input_tokens': 10, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_pause_turn_exceeds_max_continuations(allow_model_requests: None):
    """Test that exceeding the default max continuations (50) raises UnexpectedModelBehavior."""
    responses: list[BetaMessage | Exception] = []
    for _ in range(51):
        c = completion_message([BetaTextBlock(text='paused', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
        c.stop_reason = 'pause_turn'
        responses.append(c)

    mock_client = MockAnthropic.create_mock(responses)
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum continuations \\(50\\)'):
        await agent.run('test prompt', usage_limits=UsageLimits(request_limit=None))


@pytest.mark.vcr()
async def test_pause_turn_web_search_vcr(allow_model_requests: None, anthropic_api_key: str):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 4096}, max_tokens=15000)
    agent = Agent(model, builtin_tools=[WebSearchTool()], model_settings=settings)

    prompt = (
        'Run a series of web searches to gather up-to-date context. '
        'Do 6 separate searches, one at a time, and do not answer until all searches are complete. '
        'Queries: '
        '1) "San Francisco weather today", '
        '2) "San Francisco sunrise time today", '
        '3) "Golden Gate Bridge traffic today", '
        '4) "San Francisco air quality today", '
        '5) "San Francisco events this week", '
        '6) "San Francisco ferry schedule today". '
        '7) "prevailing information on quantum computing today", '
        '8) "latest news on the stock market today", '
        '9) "latest news on the weather in San Francisco today", '
        '10) "latest news on the traffic in San Francisco today", '
        '11) "latest news on the air quality in San Francisco today", '
        '12) "latest news on the events in San Francisco this week", '
        '13) "latest news on the ferry schedule in San Francisco today", '
        '14) "latest news on the quantum computing in San Francisco today", '
        '15) "latest news on the stock market in San Francisco today", '
        'After the searches, provide a concise summary.'
    )

    result = await agent.run(prompt)

    # With ContinueRequestNode, pause_turn responses are merged into the final response
    # and no longer appear as separate messages in the history.
    # Verify the agent completed successfully (which exercises the continuation path).
    assert result.output


async def test_async_request_prompt_caching(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='world', type='text')],
        usage=BetaUsage(
            input_tokens=3,
            output_tokens=5,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=6,
        ),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=13,
            cache_write_tokens=4,
            cache_read_tokens=6,
            output_tokens=5,
            details={
                'input_tokens': 3,
                'output_tokens': 5,
                'cache_creation_input_tokens': 4,
                'cache_read_input_tokens': 6,
            },
        )
    )
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    assert last_message.cost().total_price == snapshot(Decimal('0.00002688'))


async def test_cache_point_adds_cache_control(allow_model_requests: None):
    """Test that CachePoint correctly adds cache_control to content blocks.

    By default, CachePoint uses ttl='5m'.
    """
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with CachePoint after text content (default ttl='5m')
    await agent.run(['Some context to cache', CachePoint(), 'Now the question'])

    # Verify cache_control was added with default ttl='5m'
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_cache_point_multiple_markers(allow_model_requests: None):
    """Test multiple CachePoint markers in a single prompt."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    await agent.run(['First chunk', CachePoint(), 'Second chunk', CachePoint(), 'Question'])

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']

    # Default ttl='5m'
    assert content == snapshot(
        [
            {'text': 'First chunk', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'text': 'Second chunk', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'text': 'Question', 'type': 'text'},
        ]
    )


async def test_cache_point_as_first_content_raises_error(allow_model_requests: None):
    """Test that CachePoint as first content raises UserError."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(
        UserError,
        match='CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to.',
    ):
        await agent.run([CachePoint(), 'This should fail'])


async def test_cache_point_with_image_content(allow_model_requests: None):
    """Test CachePoint works with image content."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    await agent.run(
        [
            ImageUrl('https://example.com/image.jpg'),
            CachePoint(),
            'What is in this image?',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']

    # Default ttl='5m'
    assert content == snapshot(
        [
            {
                'source': {'type': 'url', 'url': 'https://example.com/image.jpg'},
                'type': 'image',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {'text': 'What is in this image?', 'type': 'text'},
        ]
    )


async def test_cache_point_in_otel_message_parts(allow_model_requests: None):
    """Test that CachePoint is handled correctly in otel message parts conversion."""
    from pydantic_ai.agent import InstrumentationSettings
    from pydantic_ai.messages import UserPromptPart

    # Create a UserPromptPart with CachePoint
    part = UserPromptPart(content=['text before', CachePoint(), 'text after'])

    # Convert to otel message parts
    settings = InstrumentationSettings(include_content=True)
    otel_parts = part.otel_message_parts(settings)

    # Should have 2 text parts, CachePoint is skipped
    assert otel_parts == snapshot(
        [{'type': 'text', 'content': 'text before'}, {'type': 'text', 'content': 'text after'}]
    )


def test_cache_control_unsupported_param_type():
    """Test that cache control raises error for unsupported param types."""
    from unittest.mock import MagicMock

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.anthropic import AnthropicModel

    # Create a mock model instance
    mock_client = MagicMock()
    mock_client.__class__.__name__ = 'AsyncAnthropic'
    mock_client.base_url = 'https://api.anthropic.com'
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Create a list with an unsupported param type (thinking)
    params: list[dict[str, Any]] = [{'type': 'thinking', 'source': {'data': 'test'}}]

    with pytest.raises(UserError, match='Cache control not supported for param type: thinking'):
        m._add_cache_control_to_last_param(params)  # type: ignore[arg-type]  # Testing internal method


def test_build_cache_control_bedrock_includes_ttl():
    """Test that _build_cache_control includes TTL for Bedrock clients."""
    from unittest.mock import MagicMock

    from anthropic import AsyncAnthropicBedrock

    # Create a mock client using spec=AsyncAnthropicBedrock for isinstance check
    mock_bedrock_client = MagicMock(spec=AsyncAnthropicBedrock)
    mock_bedrock_client.base_url = 'https://bedrock.amazonaws.com'

    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_bedrock_client))

    # Verify cache_control includes TTL for Bedrock
    cache_control = m._build_cache_control('5m')  # pyright: ignore[reportPrivateUsage]
    assert cache_control == {'type': 'ephemeral', 'ttl': '5m'}

    cache_control_1h = m._build_cache_control('1h')  # pyright: ignore[reportPrivateUsage]
    assert cache_control_1h == {'type': 'ephemeral', 'ttl': '1h'}


def test_build_cache_control_standard_client_includes_ttl():
    """Test that _build_cache_control includes TTL for standard Anthropic clients."""
    from unittest.mock import MagicMock

    # Create a mock client that looks like standard AsyncAnthropic
    mock_client = MagicMock()
    mock_client.__class__.__name__ = 'AsyncAnthropic'
    mock_client.base_url = 'https://api.anthropic.com'

    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Verify cache_control includes TTL for standard clients
    cache_control = m._build_cache_control('5m')  # pyright: ignore[reportPrivateUsage]
    assert cache_control == {'type': 'ephemeral', 'ttl': '5m'}

    cache_control_1h = m._build_cache_control('1h')  # pyright: ignore[reportPrivateUsage]
    assert cache_control_1h == {'type': 'ephemeral', 'ttl': '1h'}


async def test_cache_point_with_5m_ttl(allow_model_requests: None):
    """Test that CachePoint with explicit ttl='5m' includes the ttl field."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with explicit CachePoint(ttl='5m')
    await agent.run(['Some context to cache', CachePoint(ttl='5m'), 'Now the question'])

    # Verify cache_control was added with 5m ttl
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_cache_point_with_1h_ttl(allow_model_requests: None):
    """Test that CachePoint with ttl='1h' correctly sets the TTL."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with CachePoint(ttl='1h')
    await agent.run(['Some context to cache', CachePoint(ttl='1h'), 'Now the question'])

    # Verify cache_control was added with 1h ttl
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_anthropic_cache_tools(allow_model_requests: None):
    """Test that anthropic_cache_tool_definitions adds cache_control to last tool."""
    c = completion_message(
        [BetaTextBlock(text='Tool result', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='Test system prompt',
        model_settings=AnthropicModelSettings(anthropic_cache_tool_definitions=True),
    )

    @agent.tool_plain
    def tool_one() -> str:  # pragma: no cover
        return 'one'

    @agent.tool_plain
    def tool_two() -> str:  # pragma: no cover
        return 'two'

    await agent.run('test prompt')

    # Verify cache_control was added to the last tool
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    has_strict_tools = any('strict' in tool for tool in tools)  # we only ever set strict: True
    assert has_strict_tools is False  # ensure strict is not set for haiku-4-5
    assert tools == snapshot(
        [
            {
                'name': 'tool_one',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
            },
            {
                'name': 'tool_two',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
        ]
    )


async def test_anthropic_eager_input_streaming(allow_model_requests: None):
    """Test that anthropic_eager_input_streaming sets eager_input_streaming on all tools."""
    c = completion_message(
        [BetaTextBlock(text='Tool result', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='Test system prompt',
        model_settings=AnthropicModelSettings(anthropic_eager_input_streaming=True),
    )

    @agent.tool_plain
    def tool_one() -> str:  # pragma: no cover
        return 'one'

    @agent.tool_plain
    def tool_two() -> str:  # pragma: no cover
        return 'two'

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    assert tools == snapshot(
        [
            {
                'name': 'tool_one',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                'eager_input_streaming': True,
            },
            {
                'name': 'tool_two',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                'eager_input_streaming': True,
            },
        ]
    )


async def test_anthropic_cache_instructions(allow_model_requests: None):
    """Test that anthropic_cache_instructions adds cache_control to system prompt."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='This is a test system prompt with instructions.',
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    await agent.run('test prompt')

    # Verify system is a list with cache_control on last block
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'This is a test system prompt with instructions.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )


async def test_anthropic_cache_tools_and_instructions(allow_model_requests: None):
    """Test that both cache settings work together."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_tool_definitions=True,
            anthropic_cache_instructions=True,
        ),
    )

    @agent.tool_plain
    def my_tool(value: str) -> str:  # pragma: no cover
        return f'Result: {value}'

    await agent.run('test prompt')

    # Verify both have cache_control
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    system = completion_kwargs['system']
    has_strict_tools = any('strict' in tool for tool in tools)  # we only ever set strict: True
    assert has_strict_tools is False  # ensure strict is not set for haiku-4-5
    assert tools == snapshot(
        [
            {
                'name': 'my_tool',
                'description': '',
                'input_schema': {
                    'additionalProperties': False,
                    'properties': {'value': {'type': 'string'}},
                    'required': ['value'],
                    'type': 'object',
                },
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )
    assert system == snapshot(
        [{'type': 'text', 'text': 'System instructions to cache.', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    )


async def test_anthropic_cache_with_custom_ttl(allow_model_requests: None):
    """Test that cache settings support custom TTL values ('5m' or '1h')."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_tool_definitions='1h',  # Custom 1h TTL
            anthropic_cache_instructions='5m',  # Explicit 5m TTL
        ),
    )

    @agent.tool_plain
    def my_tool(value: str) -> str:  # pragma: no cover
        return f'Result: {value}'

    await agent.run('test prompt')

    # Verify custom TTL values are applied
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    system = completion_kwargs['system']

    # Tool definitions should have 1h TTL
    assert tools[0]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})
    # System instructions should have 5m TTL
    assert system[0]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '5m'})


async def test_anthropic_cache_instructions_mixed_static_dynamic(allow_model_requests: None):
    """Test that cache_control is placed after the last static instruction when mixed with dynamic."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        instructions='Static instructions that should be cached.',
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_context() -> str:
        return 'Dynamic context that changes per run.'

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    # Should have 2 blocks: static (with cache_control) and dynamic (without)
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'Static instructions that should be cached.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {
                'type': 'text',
                'text': 'Dynamic context that changes per run.',
            },
        ]
    )


async def test_anthropic_cache_instructions_all_dynamic(allow_model_requests: None):
    """Test that when all instructions are dynamic, no cache_control is placed on instructions."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='A static system prompt.',
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'Dynamic instructions only.'

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    # System prompt block gets cache_control (it's static), dynamic instructions don't
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'A static system prompt.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {
                'type': 'text',
                'text': 'Dynamic instructions only.',
            },
        ]
    )


async def test_anthropic_cache_instructions_all_static_with_toolset(allow_model_requests: None):
    """Test all-static cache placement with multiple instruction sources."""
    from pydantic_ai import FunctionToolset

    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    toolset = FunctionToolset(instructions='Use the tools wisely.')
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        toolsets=[toolset],
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    # All instruction parts are static — cache_control on the last block
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'You are a helpful assistant.',
            },
            {
                'type': 'text',
                'text': 'Use the tools wisely.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
        ]
    )


async def test_anthropic_cache_instructions_all_dynamic_no_system_prompt(allow_model_requests: None):
    """Test all-dynamic instructions with no system prompt — no cache_control at all."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    @agent.instructions
    def dynamic_only() -> str:
        return 'Dynamic instructions.'

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    # No system prompt, all dynamic → no cache_control
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'Dynamic instructions.',
            },
        ]
    )


async def test_anthropic_cache_instructions_no_instructions(allow_model_requests: None):
    """Test that cache_instructions with no actual instructions works gracefully."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    await agent.run('test prompt')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    # No system prompts or instructions — system param is omitted
    assert 'system' not in completion_kwargs or not completion_kwargs['system']


async def test_anthropic_incompatible_schema_disables_auto_strict(allow_model_requests: None):
    """Ensure strict mode is disabled when Anthropic cannot enforce the tool schema."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        usage=BetaUsage(input_tokens=8, output_tokens=3),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    def constrained_tool(value: Annotated[str, Field(min_length=2)]) -> str:  # pragma: no cover
        return value

    await agent.run('hello')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'strict' not in completion_kwargs['tools'][0]


async def test_beta_header_merge_builtin_tools_and_native_output(allow_model_requests: None):
    """Verify beta headers merge from custom headers, builtin tools, and native output."""
    c = completion_message(
        [BetaTextBlock(text='{"city": "Mexico City", "country": "Mexico"}', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2'}),
    )

    agent = Agent(
        model,
        builtin_tools=[MemoryTool()],
        output_type=NativeOutput(CityLocation),
    )

    @agent.tool_plain
    def memory(**command: Any) -> Any:  # pragma: no cover
        return 'memory response'

    await agent.run('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    betas = completion_kwargs['betas']

    assert betas == snapshot(
        [
            'context-management-2025-06-27',
            'custom-feature-1',
            'custom-feature-2',
            'structured-outputs-2025-11-13',
        ]
    )


async def test_model_settings_reusable_with_beta_headers(allow_model_requests: None):
    """Verify that model_settings with extra_headers can be reused across multiple runs.

    This test ensures that the beta header extraction doesn't mutate the original model_settings,
    allowing the same settings to be used for multiple agent runs.
    """
    c = completion_message(
        [BetaTextBlock(text='Hello!', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    model_settings = AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2'})

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=model_settings,
    )

    agent = Agent(model)

    # First run
    await agent.run('Hello')

    # Verify the original model_settings is not mutated
    assert model_settings.get('extra_headers') == {'anthropic-beta': 'custom-feature-1, custom-feature-2'}

    # Second run should work with the same beta headers
    await agent.run('Hello again')

    # Verify again after second run
    assert model_settings.get('extra_headers') == {'anthropic-beta': 'custom-feature-1, custom-feature-2'}

    # Verify both runs had the correct betas
    all_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(all_kwargs) == 2
    for completion_kwargs in all_kwargs:
        betas = completion_kwargs['betas']
        assert 'custom-feature-1' in betas
        assert 'custom-feature-2' in betas


async def test_anthropic_betas_setting(allow_model_requests: None):
    """Verify anthropic_betas setting adds betas to the API request."""
    c = completion_message(
        [BetaTextBlock(text='Hello!', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=AnthropicModelSettings(
            anthropic_betas=['interleaved-thinking-2025-05-14'],
        ),
    )
    agent = Agent(model)

    await agent.run('Hello')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    betas = completion_kwargs['betas']
    assert 'interleaved-thinking-2025-05-14' in betas


async def test_anthropic_betas_merge_with_other_sources(allow_model_requests: None):
    """Verify anthropic_betas merges with auto-added betas and extra_headers anthropic-beta."""
    c = completion_message(
        [BetaTextBlock(text='{"city": "Paris", "country": "France"}', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    class CityLocation(BaseModel):
        city: str
        country: str

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=AnthropicModelSettings(
            anthropic_betas=['interleaved-thinking-2025-05-14'],
            extra_headers={'anthropic-beta': 'custom-feature-1'},
        ),
    )
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    await agent.run('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    betas = completion_kwargs['betas']
    assert 'interleaved-thinking-2025-05-14' in betas
    assert 'custom-feature-1' in betas
    assert 'structured-outputs-2025-11-13' in betas


async def test_anthropic_mixed_strict_tool_run(allow_model_requests: None, anthropic_api_key: str):
    """Exercise both strict=True and strict=False tool definitions against the live API."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        system_prompt='Always call `country_source` first, then call `capital_lookup` with that result before replying.',
    )

    @agent.tool_plain(strict=True)
    async def country_source() -> str:
        return 'Japan'

    capital_called = {'value': False}

    @agent.tool_plain(strict=False)
    async def capital_lookup(country: str) -> str:
        capital_called['value'] = True
        if country == 'Japan':
            return 'Tokyo'
        return f'Unknown capital for {country}'  # pragma: no cover

    result = await agent.run('Use the registered tools and respond exactly as `Capital: <city>`.')
    assert capital_called['value'] is True
    assert result.output.startswith('Capital:')
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == 'capital_lookup'
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


async def test_anthropic_cache_messages(allow_model_requests: None):
    """Test that anthropic_cache_messages caches only the last message."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,
        ),
    )

    await agent.run('User message')

    # Verify only last message has cache_control, not system
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    messages = completion_kwargs['messages']

    # System should NOT have cache_control (should be a plain string)
    assert system == snapshot('System instructions to cache.')

    # Last message content should have cache_control
    assert messages[-1]['content'][-1] == snapshot(
        {'type': 'text', 'text': 'User message', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
    )


async def test_anthropic_cache_messages_with_custom_ttl(allow_model_requests: None):
    """Test that anthropic_cache_messages supports custom TTL values."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages='1h',  # Custom 1h TTL
        ),
    )

    await agent.run('User message')

    # Verify use 1h TTL
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    assert messages[-1]['content'][-1]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})


async def test_limit_cache_points_with_cache_messages(allow_model_requests: None):
    """Test that cache points are limited when using cache_messages + CachePoint markers."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,  # Uses 1 cache point
        ),
    )

    # Add 4 CachePoint markers (total would be 5: 1 from cache_messages + 4 from markers)
    # Only 3 CachePoint markers should be kept (newest ones)
    await agent.run(
        [
            'Context 1',
            CachePoint(),  # Oldest, should be removed
            'Context 2',
            CachePoint(),  # Should be kept
            'Context 3',
            CachePoint(),  # Should be kept
            'Context 4',
            CachePoint(),  # Should be kept
            'Question',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    # Count cache_control occurrences in messages
    cache_count = 0
    for msg in messages:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    # anthropic_cache_messages uses 1 cache point (last message only)
    # With 4 CachePoint markers, we'd have 5 total
    # Limit is 4, so 1 oldest CachePoint should be removed
    # Result: 3 cache points from CachePoint markers + 1 from cache_messages = 4 total
    assert cache_count == 4


async def test_limit_cache_points_all_settings(allow_model_requests: None):
    """Test cache point limiting with all cache settings enabled."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_instructions=True,  # 1 cache point
            anthropic_cache_tool_definitions=True,  # 1 cache point
        ),
    )

    @agent.tool_plain
    def my_tool() -> str:  # pragma: no cover
        return 'result'

    # Add 3 CachePoint markers (total would be 5: 2 from settings + 3 from markers)
    # Only 2 CachePoint markers should be kept
    await agent.run(
        [
            'Context 1',
            CachePoint(),  # Oldest, should be removed
            'Context 2',
            CachePoint(),  # Should be kept
            'Context 3',
            CachePoint(),  # Should be kept
            'Question',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    # Count cache_control in messages (excluding system and tools)
    cache_count = 0
    for msg in messages:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    # Should have exactly 2 cache points in messages
    # (4 total - 1 system - 1 tool = 2 available for messages)
    assert cache_count == 2


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='world', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=3,
            output_tokens=5,
            details={'input_tokens': 3, 'output_tokens': 5},
        )
    )


async def test_request_stream_fallback_for_high_max_tokens(
    allow_model_requests: None, anthropic_api_key: str, vcr: Any
):
    """When the Anthropic SDK raises ValueError for high max_tokens, request() falls back to streaming."""
    # https://github.com/anthropics/anthropic-sdk-python/blob/49d639a671cb0ac30c767e8e1e68fdd5925205d5/src/anthropic/_base_client.py#L726
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        'What is 1+1? Answer with just the number.', model_settings=ModelSettings(max_tokens=32_000)
    )

    # Verify the fallback used streaming — the only request recorded should have stream=true
    assert len(vcr.requests) == 1
    request_body = json.loads(vcr.requests[0].body)
    assert request_body['stream'] is True
    assert request_body['max_tokens'] == 32_000

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 1+1? Answer with just the number.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='2')],
                usage=RequestUsage(
                    input_tokens=20,
                    output_tokens=5,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 20,
                        'output_tokens': 5,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot('2')


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [BetaToolUseBlock(id='123', input={'response': [1, 2, 3]}, name='final_result', type='tool_use')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('hello')
    assert result.output == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': [1, 2, 3]},
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            [BetaToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [BetaToolUseBlock(id='2', input={'loc_name': 'London'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=3, output_tokens=2),
        ),
        completion_message(
            [BetaTextBlock(text='final response', type='text')],
            usage=BetaUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, instructions='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'San Francisco'},
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1, details={'input_tokens': 2, 'output_tokens': 1}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'London'},
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=2, details={'input_tokens': 3, 'output_tokens': 2}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def get_mock_chat_completion_kwargs(async_anthropic: AsyncAnthropic) -> list[dict[str, Any]]:
    if isinstance(async_anthropic, MockAnthropic):
        return async_anthropic.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockOpenAI instance')


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    responses = [
        completion_message(
            [BetaToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [BetaTextBlock(text='final response', type='text')],
            usage=BetaUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})  # pragma: no cover
        else:
            raise ModelRetry('Wrong location, please try again')

    await agent.run('hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['tool_choice']['disable_parallel_tool_use'] == (
        not parallel_tool_calls
    )


async def test_multiple_parallel_tool_calls(allow_model_requests: None):
    async def retrieve_entity_info(name: str) -> str:
        """Get the knowledge about the given entity."""
        data = {
            'alice': "alice is bob's wife",
            'bob': "bob is alice's husband",
            'charlie': "charlie is alice's son",
            'daisy': "daisy is bob's daughter and charlie's younger sister",
        }
        return data[name.lower()]

    system_prompt = """
    Use the `retrieve_entity_info` tool to get information about a specific person.
    If you need to use `retrieve_entity_info` to get information about multiple people, try
    to call them in parallel as much as possible.
    Think step by step and then provide a single most probable concise answer.
    """

    # If we don't provide some value for the API key, the anthropic SDK will raise an error.
    # However, we do want to use the environment variable if present when rewriting VCR cassettes.
    api_key = os.getenv('ANTHROPIC_API_KEY', 'mock-value')
    agent = Agent(
        AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=api_key)),
        system_prompt=system_prompt,
        tools=[retrieve_entity_info],
    )

    result = await agent.run('Alice, Bob, Charlie and Daisy are a family. Who is the youngest?')
    assert 'Daisy is the youngest' in result.output

    all_messages = result.all_messages()
    first_response = all_messages[1]
    second_request = all_messages[2]
    assert first_response.parts == snapshot(
        [
            TextPart(
                content="I'll help you find out who is the youngest by retrieving information about each family member. I'll retrieve their entity information to compare their ages.",
                part_kind='text',
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Alice'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Bob'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Charlie'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Daisy'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
        ]
    )
    assert second_request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="alice is bob's wife",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="bob is alice's husband",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="charlie is alice's son",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="daisy is bob's daughter and charlie's younger sister",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
        ]
    )

    # Ensure the tool call IDs match between the tool calls and the tool returns
    tool_call_part_ids = [part.tool_call_id for part in first_response.parts if part.part_kind == 'tool-call']
    tool_return_part_ids = [part.tool_call_id for part in second_request.parts if part.part_kind == 'tool-return']
    assert len(set(tool_call_part_ids)) == 4  # ensure they are all unique
    assert tool_call_part_ids == tool_return_part_ids


async def test_anthropic_specific_metadata(allow_model_requests: None) -> None:
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello', model_settings=AnthropicModelSettings(anthropic_metadata={'user_id': '123'}))
    assert result.output == 'world'
    assert get_mock_chat_completion_kwargs(mock_client)[0]['metadata']['user_id'] == '123'


async def test_stream_structured(allow_model_requests: None):
    """Test streaming structured responses with Anthropic's API.

    This test simulates how Anthropic streams tool calls:
    1. Message start
    2. Tool block start with initial data
    3. Tool block delta with additional data
    4. Tool block stop
    5. Update usage
    6. Message stop
    """
    stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=20, output_tokens=0),
            ),
        ),
        # Start tool block with initial data
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaToolUseBlock(type='tool_use', id='tool_1', name='my_tool', input={}),
        ),
        # Add more data through an incomplete JSON delta
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='{"first": "One'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='", "second": "Two"'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='}'),
        ),
        # Mark tool block as complete
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        # Update the top-level message with usage
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=20, output_tokens=5),
        ),
        # Mark message as complete
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    done_stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=0, output_tokens=0),
            ),
        ),
        # Text block with final data
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(type='text', text='FINAL_PAYLOAD'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=0, output_tokens=0),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock([stream, done_stream])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    tool_called = False

    @agent.tool_plain
    async def my_tool(first: str, second: str) -> int:
        nonlocal tool_called
        tool_called = True
        return len(first) + len(second)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        chunks = [c async for c in result.stream_output(debounce_by=None)]

        # The tool output doesn't echo any content to the stream, so we only get the final payload once when
        # the block starts and once when it ends.
        assert chunks == snapshot(['FINAL_PAYLOAD', 'FINAL_PAYLOAD'])
        assert result.is_complete
        assert result.usage() == snapshot(
            RunUsage(
                requests=2,
                input_tokens=20,
                output_tokens=5,
                tool_calls=1,
                details={'input_tokens': 20, 'output_tokens': 5},
            )
        )
        assert tool_called
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='FINAL_PAYLOAD')],
                        usage=RequestUsage(details={'input_tokens': 0, 'output_tokens': 0}),
                        model_name='claude-3-5-haiku-123',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                        provider_url='https://api.anthropic.com',
                        provider_details={'finish_reason': 'end_turn'},
                        provider_response_id='msg_123',
                        finish_reason='stop',
                    )
                )


async def test_text_content_input(allow_model_requests: None, anthropic_api_key: str):
    """Test that _map_message correctly maps a user message with TextContent."""
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System instructions.'),
                UserPromptPart(
                    content=[
                        'Hello Pydantic AI!',
                        TextContent(content='This is awesome', metadata={'font': 'bold'}),
                        TextContent(content='', metadata={'font': 'italic'}),  # Empty content would be filtered out
                    ]
                ),
            ],
        ),
        ModelResponse(
            parts=[TextPart(content='Hello Human!')],
        ),
    ]
    m = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        messages,
        ModelRequestParameters(),
        {},
    )
    assert m == snapshot(
        (
            'System instructions.',
            [
                {
                    'role': 'user',
                    'content': [
                        {'text': 'Hello Pydantic AI!', 'type': 'text'},
                        {'text': 'This is awesome', 'type': 'text'},
                    ],
                },
                {'role': 'assistant', 'content': [{'text': 'Hello Human!', 'type': 'text'}]},
            ],
        )
    )


async def test_image_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        "This is a potato. It's a yellow/golden-colored potato with a smooth, slightly bumpy skin typical of many potato varieties. The potato appears to be a whole, unpeeled tuber with a classic oblong or oval shape. Potatoes are starchy root vegetables that are widely consumed around the world and can be prepared in many ways, such as boiling, baking, frying, or mashing."
    )


async def test_image_url_input_force_download(
    allow_model_requests: None, anthropic_api_key: str, disable_ssrf_protection_for_vcr: None
):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(
                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                force_download=True,
            ),
        ]
    )
    assert result.output == snapshot(
        """\
This is a **potato**, specifically a yellow or gold potato variety. You can identify it by its characteristic features:

- **Oval/round shape** with smooth skin
- **Golden-yellow color** with small dark spots or eyes
- **Starchy appearance** typical of potatoes

This appears to be a russet or similar yellow potato variety commonly used for cooking, baking, or making mashed potatoes.\
"""
    )


async def test_extra_headers(allow_model_requests: None, anthropic_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        model_settings=AnthropicModelSettings(
            anthropic_metadata={'user_id': '123'}, extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}
        ),
    )
    await agent.run('hello')


async def test_image_url_input_invalid_mime_type(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What animal is this?',
            ImageUrl(
                url='https://lh3.googleusercontent.com/proxy/YngsuS8jQJysXxeucAgVBcSgIdwZlSQ-HvsNxGjHS0SrUKXI161bNKh6SOcMsNUGsnxoOrS3AYX--MT4T3S3SoCgSD1xKrtBwwItcgexaX_7W-qHo-VupmYgjjzWO-BuORLp9-pj8Kjr'
            ),
        ]
    )
    assert result.output == snapshot(
        'This is a Great Horned Owl (Bubo virginianus), a large and powerful owl species native to the Americas. The image shows the owl perched on a log or branch, surrounded by soft yellow and green vegetation. The owl has distinctive ear tufts (the "horns"), large yellow eyes, and a mottled gray-brown plumage that provides excellent camouflage in woodland and grassland environments. Great Horned Owls are known for their impressive size, sharp talons, and nocturnal hunting habits. They are formidable predators that can hunt animals as large as skunks, rabbits, and even other birds of prey.'
    )


async def test_image_url_force_download() -> None:
    """Test that force_download=True calls download_item for ImageUrl."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': b'\x89PNG\r\n\x1a\n fake image data',
            'content_type': 'image/png',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/image.png'


async def test_image_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for ImageUrl."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_not_called()


async def test_document_url_pdf_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl (PDF)."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': b'%PDF-1.4 fake pdf data',
            'content_type': 'application/pdf',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/doc.pdf',
                                media_type='application/pdf',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/doc.pdf'


async def test_document_url_text_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl (text/plain)."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'Sample text content',
            'content_type': 'text/plain',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test text file',
                            DocumentUrl(
                                url='https://example.com/doc.txt',
                                media_type='text/plain',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/doc.txt'


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot(
        "status_code: 500, model_name: claude-sonnet-4-5, body: {'error': 'test error'}"
    )


def test_model_connection_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIConnectionError(
            message='Connection to https://api.anthropic.com timed out',
            request=httpx.Request('POST', 'https://api.anthropic.com/v1/messages'),
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'claude-sonnet-4-5'
    assert 'Connection to https://api.anthropic.com timed out' in str(exc_info.value.message)


async def test_count_tokens_connection_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIConnectionError(
            message='Connection to https://api.anthropic.com timed out',
            request=httpx.Request('POST', 'https://api.anthropic.com/v1/messages'),
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))
    assert exc_info.value.model_name == 'claude-sonnet-4-5'
    assert 'Connection to https://api.anthropic.com timed out' in str(exc_info.value.message)


async def test_document_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, document_content: BinaryContent
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', document_content])
    assert result.output == snapshot(
        'The document simply contains the text "Dummy PDF file" at the top of what appears to be an otherwise blank page.'
    )


async def test_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'This document appears to be a sample PDF file that mainly contains Lorem ipsum text, which is placeholder text commonly used in design and publishing. The document starts with "Sample PDF" as its title, followed by the line "This is a simple PDF file. Fun fun fun." The rest of the content consists of several paragraphs of Lorem ipsum text, which is Latin-looking but essentially meaningless text used to demonstrate the visual form of a document without the distraction of meaningful content.'
    )


async def test_text_document_url_input(
    allow_model_requests: None, anthropic_api_key: str, disable_ssrf_protection_for_vcr: None
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
This document is a TXT test file that contains example content about the use of placeholder names like "John Doe," "Jane Doe," and their variants in legal and cultural contexts. The main content is divided into three main paragraphs explaining:

1. The use of "Doe" names as placeholders for unknown parties in legal actions
2. The use of "John Doe" as a reference to a typical male in various contexts
3. The use of variations like "Baby Doe" and numbered "John Doe"s in specific cases

The document also includes metadata about the file itself, including its purpose, type, and version, as well as attribution information indicating that the example content is from Wikipedia and is licensed under Attribution-ShareAlike 4.0.\
""")


async def test_text_document_as_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, text_document_content: BinaryContent
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What does this text file say?', text_document_content])
    assert result.output == snapshot('The text file says "Dummy TXT file".')


async def test_uploaded_file_with_text(allow_model_requests: None) -> None:
    """Test that UploadedFile is correctly mapped to a document block with file source."""
    c = completion_message(
        [BetaTextBlock(text='The file contains important data.', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=8),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run(['Analyze this file', UploadedFile(file_id='file-abc123', provider_name='anthropic')])

    assert result.output == 'The file contains important data.'

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Analyze this file', 'type': 'text'},
                    {'source': {'file_id': 'file-abc123', 'type': 'file'}, 'type': 'document'},
                ],
            }
        ]
    )


async def test_uploaded_file_only(allow_model_requests: None) -> None:
    """Test UploadedFile as the only content in a message."""
    c = completion_message(
        [BetaTextBlock(text='This is a PDF document.', type='text')],
        usage=BetaUsage(input_tokens=5, output_tokens=6),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run([UploadedFile(file_id='file-xyz789', provider_name='anthropic')])

    assert result.output == 'This is a PDF document.'

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']
    assert content == snapshot([{'source': {'file_id': 'file-xyz789', 'type': 'file'}, 'type': 'document'}])


async def test_multiple_uploaded_files(allow_model_requests: None) -> None:
    """Test multiple UploadedFiles in a single message."""
    c = completion_message(
        [BetaTextBlock(text='Both files contain similar data.', type='text')],
        usage=BetaUsage(input_tokens=15, output_tokens=7),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'Compare these files',
            UploadedFile(file_id='file-001', provider_name='anthropic'),
            UploadedFile(file_id='file-002', provider_name='anthropic'),
        ]
    )

    assert result.output == 'Both files contain similar data.'

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']
    assert content == snapshot(
        [
            {'text': 'Compare these files', 'type': 'text'},
            {'source': {'file_id': 'file-001', 'type': 'file'}, 'type': 'document'},
            {'source': {'file_id': 'file-002', 'type': 'file'}, 'type': 'document'},
        ]
    )


async def test_uploaded_file_image(allow_model_requests: None) -> None:
    """Test that UploadedFile with image media type is mapped to an image block."""
    c = completion_message(
        [BetaTextBlock(text='The image shows a cat.', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=6),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        ['Describe this image', UploadedFile(file_id='file-img123', provider_name='anthropic', media_type='image/png')]
    )

    assert result.output == 'The image shows a cat.'

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'Describe this image', 'type': 'text'},
                    {'source': {'file_id': 'file-img123', 'type': 'file'}, 'type': 'image'},
                ],
            }
        ]
    )


async def test_uploaded_file_wrong_provider(allow_model_requests: None) -> None:
    """Test that UploadedFile with wrong provider raises an error."""
    c = completion_message(
        [BetaTextBlock(text='Should not reach here.', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=8),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(UserError, match="provider_name='openai'.*cannot be used with AnthropicModel"):
        await agent.run(['Analyze this file', UploadedFile(file_id='file-abc123', provider_name='openai')])


async def test_uploaded_file_unsupported_media_type(allow_model_requests: None) -> None:
    """Test that UploadedFile with unsupported media type (e.g. audio) raises an error."""
    c = completion_message(
        [BetaTextBlock(text='Should not reach here.', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=8),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(UserError, match='Unsupported media type.*audio/mpeg'):
        await agent.run(
            [
                'Analyze this file',
                UploadedFile(file_id='file-abc123', provider_name='anthropic', media_type='audio/mpeg'),
            ]
        )


def test_init_with_provider():
    provider = AnthropicProvider(api_key='api-key')
    model = AnthropicModel('claude-3-opus-latest', provider=provider)
    assert model.model_name == 'claude-3-opus-latest'
    assert model.client == provider.client


def test_init_with_provider_string(env: TestEnv):
    env.set('ANTHROPIC_API_KEY', 'env-api-key')
    model = AnthropicModel('claude-3-opus-latest', provider='anthropic')
    assert model.model_name == 'claude-3-opus-latest'
    assert model.client is not None


async def test_anthropic_model_instructions(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-opus-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    @agent.instructions
    def simple_instructions():
        return 'You are a helpful assistant.'

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(
                    input_tokens=20,
                    output_tokens=10,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 20,
                        'output_tokens': 10,
                    },
                ),
                model_name='claude-3-opus-20240229',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Fg1JVgvCYUHWsxrj9GkpEv',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='This is a straightforward question about pedestrian safety. I should provide clear, practical advice about crossing the street safely.',
                        signature='Eq8CCkYICxgCKkDdadQOXMzNBqjrNVAKWsgUfg49NpPg026zBGxIIGwEHVCq0JTW/P9fKHjZdjgO8Dyx03YDw6hN0w1HucifXFggEgzoVS5Gogi9nvJSOA8aDDYuCAX4nGGkeHQLayIw+MWbf/TYU4AqT1X89p4S7fe7LOO+B8o24yCHQ8cFK9QK9p5WMj2Y4oBFBfC9uL8ZKpYBDjoKceyqFJA56ewVH73lNY5szTvm52+CVXMZJCb8x0B1bf9LIOsFUoJD6F4gZBdKfMqJgFCcKFR6iZh09pwa0E8lHvEnUeF1A0AJ6z0j8gQd5NxgipxWrF9908qJbMSkVDdg1dT/3Rr0nbGguAYTYdoV4MrVxyk29dSkkjyAAZBMI3p+HOwiaT6GmYq4qVE3kWnSoiEJGAE=',
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=43,
                    output_tokens=321,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 43,
                        'output_tokens': 321,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01TGA8SWcHTTn5674cmicbnJ',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
This is an interesting analogy question. The person is asking me to apply the safety principles and general approach from crossing a street to crossing a river. Let me think about the parallel elements:

Street crossing principles:
- Find safe crossing point
- Assess conditions
- Look for hazards
- Use designated crossings when available
- Wait for safe conditions
- Cross carefully while staying alert

River crossing would involve similar safety thinking:
- Find safe crossing point (bridge, ford, ferry, shallow area)
- Assess conditions (water depth, current, weather)
- Look for hazards (rocks, debris, cold water, strong current)
- Use established crossings when available
- Wait for safe conditions (water levels, weather)
- Cross carefully while staying alert

I should provide practical advice for different methods of crossing a river.\
""",
                        signature='EvgHCkYICxgCKkBWep44ZkS8HkPkKt2q7OJir9S1aK8TFXpFjWz4yEEVk+2r0FCXIRwuIJBfrLI+kTWKzAFtjxpM+G+S8Btnle6sEgwSq3W1+WbBYHYolVYaDCbom89zf38EbOe8jCIwKz0NLPNu1XU3I3nREDwVSSBCe/u2C+Ryon6gXHWSWlM7r6M2jMVUNynufqiO9m+jKt8GDH5qCKJRfydyyKcS1muFqazBmHs8L3sUsHzj7s2XkvP+2yA789klS3DrrYj4H1kYbRWpmGlTxkpPAuXUr8u1U02sNS0zqh5HiIEu2LZesOj5l1jw68VXcVBPsYEdkSvarScNKzmDBOiw0vTV9EkoxZ/p/ZvoP4PUYSzFc1oJRPaLDCn7KW/aAsZBbsS55YDwHBXvjrDFFtcd2V04JuavcKi0EwomwCy95e0NAaOrA9aAFizZoG30V9KSiz0XUQ3+8ByxKILXk1qvtaV2HJgYahAuRcOpEoty4+Dqx96KsA4ifPaU0+MRwoVUwGUm+mK75ViBIAQdRFblkHbPHYHpK+P9SjdIb00h6PUH59pPyNFQOMJyav7c6dy2efTmiTdzejLHXjUzVvG2LaDnq7cFM2MpqvxlIxDULVG+N13xOTStjLJ9Siwq/zMPKTZYhbYYYC6INlMxwmvM0xz3ofsZbUVOAHv2Ti9jixmB38wyKaFiS7GkQvaK9r9AYl7b632bnsjexiHMe+HMAwfOiA9d2bfhGYCwnt59uNCPgXRihLqaeemq84tiHjSpXrYAieAHtiwEhh0Zz5/ztFgn9pDko5ZmfUvXW9kcZ/8nthmDJSD0z933gw5gITW5u+4S4ozqkGtQ4lGgHNzXLpAEs1A6lsqh2jC2iAskj4Mc/oihJbmAFT0UQ0uopcExyImY6maqKub7xYUseRiNjd1Y7hq7eLDlrMiOR8DDoUoTEIz1imI+KetpLXJoSorecGkYivZajx9ZY+L/R4VcA6olgJsjSpztEvlNextE8sAcAnwBK5l8+yBxWBflFf96wOcvbxE3xtEfR5+ISy6+A6kcxPkpj/31B0VM9y3EqMcDqKmMCF6r7MpwRzXxkHofWCG49N4SQKDJrRJSMldy/qGvd5TIVDghEK+8AoVhWZXqXl6y9z5NG72fOlXLdh3me1jtqMSBX3q0gxmljqzqii/r4F6Qmmmwl2szfryxwgUWAAPS6yDEWbDyUhQSmc24Q+uHrXhKIPKuDQljCsI2by2pyC8UV4RsEfvNLk0zs5CPR3+1kewb8TVB6S+IpmJHJnBZImkI2vt2IUgvnJb/5D+aezG2mA8O+4qjsnHsbT8Njk92tOI1wxOFSAO19SOEa+DD2bsYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=354,
                    output_tokens=525,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 354,
                        'output_tokens': 525,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_redacted(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    result = await agent.run(
        'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=92,
                    output_tokens=196,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 92,
                        'output_tokens': 196,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01TbZ1ZKNMPq28AgBLyLX3c4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'What was that?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What was that?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=168,
                    output_tokens=232,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 168,
                        'output_tokens': 232,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_012oSSVsQdwoGH6b2fryM4fF',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_redacted_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=92,
                    output_tokens=189,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 92,
                        'output_tokens': 189,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_018XZkwvj9asBiffg3fXt88s',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature=IsStr(),
                    provider_name='anthropic',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EqkECkYIBxgCKkA8AZ4noDfV5VcOJe/p3JTRB6Xz5297mrWhl3MbHSXDKTMfuB/Z52U2teiWWTN0gg4eQ4bGS9TPilFX/xWTIq9HEgyOmstSPriNwyn1G7AaDC51r0hQ062qEd55IiIwYQj3Z3MSBBv0bSVdXi60LEHDvC7tzzmpQfw5Hb6R9rtyOz/6vC/xPw9/E1mUqfBqKpADO2HS2QlE/CnuzR901nZOn0TOw7kEXwH7kg30c85b9W7iKALgEejY9sELMBdPyIZNlTgKqNOKtY3R/aV5rGIRPTHh2Wh9Ijmqsf/TT7i//Z+InaYTo6f/fxF8R0vFXMRPOBME4XIscb05HcNhh4c9FDkpqQGYKaq31IR1NNwPWA0BsvdDz7SIo1nfx4H+X0qKKqqegKnQ3ynaXiD5ydT1C4U7fku4ftgF0LGwIk4PwXBE+4BP0DcKr1HV3cn7YSyNakBSDTvRJMKcXW6hl7X3w2a4//sxjC1Cjq0uzkIHkhzRWirN0OSXt+g3m6b1ex0wGmSyuO17Ak6kgVBpxwPugtrqsflG0oujFem44hecXJ9LQNssPf4RSlcydiG8EXp/XLGTe0YfHbe3kJagkowSH/Dm6ErXBiVs7249brncyY8WA+7MOoqIM82YIU095B9frCqDJDUWnN84VwOszRrcaywmpJXZO4aeQLMC1kXD5Wabu+O/00tD/X67EWkkWuR0AhDIXXjpot45vnBd4ewJ/hgB',
                    provider_name='anthropic',
                ),
                next_part_kind='thinking',
            ),
            PartStartEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EtgBCkYIBxgCKkDQfGkwzflEJP5asG3oQfJXcTwJLoRznn8CmuczWCsJ36dv93X9H0NCeaJRbi5BrCA2DyMgFnRKRuzZx8VTv5axEgwkFmcHJk8BSiZMZRQaDDYv2KZPfbFgRa2QjyIwm47f5YYsSK9CT/oh/WWpU1HJJVHr8lrC6HG1ItRdtMvYQYmEGy+KhyfcIACfbssVKkDGv/NKqNMOAcu0bd66gJ2+R1R0PX11Jxn2Nd1JtZqkxx7vMT/PXtHDhm9jkDZ2k/6RjRRFuab/DBV3yRYdZ1J0GAE=',
                    provider_name='anthropic',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EtgBCkYIBxgCKkDQfGkwzflEJP5asG3oQfJXcTwJLoRznn8CmuczWCsJ36dv93X9H0NCeaJRbi5BrCA2DyMgFnRKRuzZx8VTv5axEgwkFmcHJk8BSiZMZRQaDDYv2KZPfbFgRa2QjyIwm47f5YYsSK9CT/oh/WWpU1HJJVHr8lrC6HG1ItRdtMvYQYmEGy+KhyfcIACfbssVKkDGv/NKqNMOAcu0bd66gJ2+R1R0PX11Jxn2Nd1JtZqkxx7vMT/PXtHDhm9jkDZ2k/6RjRRFuab/DBV3yRYdZ1J0GAE=',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=2, part=TextPart(content="I notice that you've sent what"), previous_part_kind='thinking'
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' appears to be some')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' kind of test string')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=" or command. I don't have")),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' any special "magic string"')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' triggers or backdoor commands')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' that would expose internal systems or')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' change my behavior.')),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\


I'm Claude\
"""
                ),
            ),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=', an AI assistant create')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='d by Anthropic to')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' be helpful, harmless')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=', and honest. How')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' can I assist you today with')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' a legitimate task or question?')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content="""\
I notice that you've sent what appears to be some kind of test string or command. I don't have any special "magic string" triggers or backdoor commands that would expose internal systems or change my behavior.

I'm Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. How can I assist you today with a legitimate task or question?\
"""
                ),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_from_other_model(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-5', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, instructions='You are a helpful assistant.', model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c1fdbecbf081a18085a084257a9aef06da9901a3d98ab7',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=23, output_tokens=2211, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 10, 22, 37, 27, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c1fda6f11081a1b9fa80ae9122743506da9901a3d98ab7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=AnthropicModel(
            'claude-sonnet-4-0',
            provider=AnthropicProvider(api_key=anthropic_api_key),
            settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=1343,
                    output_tokens=538,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1343,
                        'output_tokens': 538,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_016e2w8nkCuArd5HFSfEwke7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=43,
                    output_tokens=282,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 43,
                        'output_tokens': 282,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01ALwQ87pTS7hH1PjSdC9wJD',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This is basic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' safety information that could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' help prevent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' accidents.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='EvMCCkYICxgCKkCHP2cSuEdcJK/0rFwqES/ecn+VurRpNTwI4XNyM0vnNfGsc9OmE8YYHauwBZ/uaRpmlEn2I4/kszHlcpptO82JEgyRMSbPkJYaegxYF3AaDHZbSm9EzZ6CM+YtliIw3iNVP/ilYrfoneo8S2+ad/5xSC62nKbk6joLtKmqXgXwYFJRpjIUjM2V7EGReOPRKtoBKfNHVmdNf7SeMhHalX/ObSeJ1G/NjDyGQAsDjyHGd7uY1r5gAIn3Cpdv5r+gHYJmWT+w2uiKZsBDRoSf4O3Km0l752EhPD4InEhqpCKyqhbUZ3dt5+JVKQHk2iyTBhQMB/XBYgZTstIpRqQRXU5ypcrydgnqj3mD1G9C7YC0ZTCNvFluAx0OL8q+cQwufgfqKquLEf2+XMYzhx9jYkVFEpnf/s1nx6gNBATKfF3Dmrs2r4tWu2QJB+FjlRuDp/8dxUxgJbmyhGxb7XsYeb1vgb7wwzDvP/UhjfQYAQ=='
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='This is a straightforward question about pedestrian safety. I should provide clear, helpful advice about how to safely cross a street. This is basic safety information that could help prevent accidents.',
                    signature='EvMCCkYICxgCKkCHP2cSuEdcJK/0rFwqES/ecn+VurRpNTwI4XNyM0vnNfGsc9OmE8YYHauwBZ/uaRpmlEn2I4/kszHlcpptO82JEgyRMSbPkJYaegxYF3AaDHZbSm9EzZ6CM+YtliIw3iNVP/ilYrfoneo8S2+ad/5xSC62nKbk6joLtKmqXgXwYFJRpjIUjM2V7EGReOPRKtoBKfNHVmdNf7SeMhHalX/ObSeJ1G/NjDyGQAsDjyHGd7uY1r5gAIn3Cpdv5r+gHYJmWT+w2uiKZsBDRoSf4O3Km0l752EhPD4InEhqpCKyqhbUZ3dt5+JVKQHk2iyTBhQMB/XBYgZTstIpRqQRXU5ypcrydgnqj3mD1G9C7YC0ZTCNvFluAx0OL8q+cQwufgfqKquLEf2+XMYzhx9jYkVFEpnf/s1nx6gNBATKfF3Dmrs2r4tWu2QJB+FjlRuDp/8dxUxgJbmyhGxb7XsYeb1vgb7wwzDvP/UhjfQYAQ==',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Here are'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

- Stop\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the curb and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' look')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left, right')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
, then left again
-\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wait for a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gap')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 in traffic
- Walk\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' br')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iskly but')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=" don't run")),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

- Keep\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' looking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for traffic as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you cross')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


**General\
"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 safety tips:**
- Put\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' away')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' phones and remove')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' head')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
phones
-\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wear bright')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' reflective clothing in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' low light')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

- Never\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' assume')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' see you')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

-\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Avoid crossing between')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 parked cars
- Walk\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' facing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' when there')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="'s no")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sidew')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
alk

**In\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' busy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' urban')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 areas:**
- Follow\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedest')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='rian signals strictly')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

- Be\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extra cautious of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cyclists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bike')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 lanes
- Watch\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for buses')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' blind')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 spots

The\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' key is to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' be visible')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=', alert, and predict')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='able in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your movements. Always')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' prioritize safety over speed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' when')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' streets.')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
Here are the basic steps for safely crossing the street:

**At intersections with traffic lights:**
- Wait for the pedestrian "Walk" signal
- Look both ways before stepping into the street
- Make eye contact with drivers when possible
- Stay alert for turning vehicles

**At intersections without signals:**
- Use crosswalks when available
- Stop at the curb and look left, right, then left again
- Wait for a clear gap in traffic
- Walk briskly but don't run
- Keep looking for traffic as you cross

**General safety tips:**
- Put away phones and remove headphones
- Wear bright or reflective clothing in low light
- Never assume drivers see you
- Avoid crossing between parked cars
- Walk facing traffic when there's no sidewalk

**In busy urban areas:**
- Follow pedestrian signals strictly
- Be extra cautious of cyclists in bike lanes
- Watch for buses and large vehicles with blind spots

The key is to be visible, alert, and predictable in your movements. Always prioritize safety over speed when crossing streets.\
"""
                ),
            ),
        ]
    )


@pytest.mark.parametrize(
    'case_id',
    ['basic', 'effort', 'adaptive-thinking'],
)
async def test_anthropic_opus_46_features(
    allow_model_requests: None,
    anthropic_api_key: str,
    case_id: str,
):
    settings_map: dict[str, AnthropicModelSettings] = {
        'basic': AnthropicModelSettings(),
        'effort': AnthropicModelSettings(anthropic_effort='low'),
        'adaptive-thinking': AnthropicModelSettings(anthropic_thinking={'type': 'adaptive'}),
    }
    has_thinking = case_id == 'adaptive-thinking'
    m = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, model_settings=settings_map[case_id])

    result = await agent.run('What is 2+2?')
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'claude-opus-4-6'

    if has_thinking:
        assert any(isinstance(p, ThinkingPart) for p in response.parts)
    assert any(isinstance(p, TextPart) for p in response.parts)


@pytest.mark.parametrize(
    'thinking_value,expected_thinking,expected_effort',
    [
        pytest.param(True, {'type': 'adaptive'}, None, id='true-adaptive'),
        pytest.param('high', {'type': 'adaptive'}, 'high', id='high-adaptive-with-effort'),
        pytest.param('low', {'type': 'adaptive'}, 'low', id='low-adaptive-with-effort'),
        pytest.param('medium', {'type': 'adaptive'}, 'medium', id='medium-adaptive-with-effort'),
        pytest.param('xhigh', {'type': 'adaptive'}, 'max', id='xhigh-maps-to-max'),
    ],
)
async def test_anthropic_unified_thinking_adaptive_model(
    allow_model_requests: None,
    thinking_value: bool | str,
    expected_thinking: dict[str, str],
    expected_effort: str | None,
):
    """Verify that unified thinking on adaptive models sends {type: 'adaptive'} + output_config effort."""
    from anthropic._types import omit as OMIT

    responses = [
        completion_message(
            [BetaTextBlock(text='4', type='text')],
            usage=BetaUsage(input_tokens=10, output_tokens=1),
        ),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking=thinking_value))  # pyright: ignore[reportArgumentType]

    await agent.run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['thinking'] == expected_thinking

    if expected_effort is not None:
        assert kwargs['output_config'] == {'effort': expected_effort}
    else:
        assert kwargs.get('output_config') is None or kwargs['output_config'] is OMIT


async def test_anthropic_unified_thinking_non_adaptive_model(allow_model_requests: None):
    """Verify that unified thinking on non-adaptive models sends {type: 'enabled', budget_tokens: N}."""
    from anthropic._types import omit as OMIT

    responses = [
        completion_message(
            [BetaTextBlock(text='4', type='text')],
            usage=BetaUsage(input_tokens=10, output_tokens=1),
        ),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking='high'))

    await agent.run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert kwargs['thinking'] == {'type': 'enabled', 'budget_tokens': 16384}
    # Non-adaptive models don't support effort
    assert kwargs.get('output_config') is None or kwargs['output_config'] is OMIT


async def test_anthropic_unified_thinking_false_omits_param(allow_model_requests: None):
    """Verify that thinking=False does not send a thinking parameter at all."""
    from anthropic._types import omit as OMIT

    responses = [
        completion_message(
            [BetaTextBlock(text='4', type='text')],
            usage=BetaUsage(input_tokens=10, output_tokens=1),
        ),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(thinking=False))

    await agent.run('What is 2+2?')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    # thinking=False on always-thinking model is silently ignored — thinking param is OMIT
    assert kwargs.get('thinking') is OMIT


async def test_anthropic_opus_46_adaptive_thinking_rejects_tool_output(allow_model_requests: None):
    """Verified in https://logfire-us.pydantic.dev/public-trace/ca9932da-b5ff-46f0-b277-9aeecc5f41e7?spanId=15a32e26f5020e62"""
    responses = [
        completion_message(
            [BetaTextBlock(text='Paris', type='text')],
            usage=BetaUsage(input_tokens=2, output_tokens=1),
        ),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(anthropic_client=mock_client))

    class CityLocation(BaseModel):
        city: str

    agent = Agent(
        m,
        output_type=ToolOutput(CityLocation),
        model_settings=AnthropicModelSettings(anthropic_thinking={'type': 'adaptive'}),
    )
    with pytest.raises(UserError, match='Anthropic does not support thinking and output tools at the same time'):
        await agent.run('What is the capital of France?')


async def test_multiple_system_prompt_formatting(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic().create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.system_prompt
    def system_prompt() -> str:
        return 'and this is another'

    await agent.run('hello')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'system' in completion_kwargs
    assert completion_kwargs['system'] == 'this is the system prompt\n\nand this is another'


def anth_msg(usage: BetaUsage) -> BetaMessage:
    return BetaMessage(
        id='x',
        content=[],
        model='claude-sonnet-4-5',
        role='assistant',
        type='message',
        usage=usage,
    )


@pytest.mark.parametrize(
    'message_callback,usage',
    [
        pytest.param(
            lambda: anth_msg(BetaUsage(input_tokens=1, output_tokens=1)),
            snapshot(RequestUsage(input_tokens=1, output_tokens=1, details={'input_tokens': 1, 'output_tokens': 1})),
            id='AnthropicMessage',
        ),
        pytest.param(
            lambda: anth_msg(
                BetaUsage(input_tokens=1, output_tokens=1, cache_creation_input_tokens=2, cache_read_input_tokens=3)
            ),
            snapshot(
                RequestUsage(
                    input_tokens=6,
                    cache_write_tokens=2,
                    cache_read_tokens=3,
                    output_tokens=1,
                    details={
                        'cache_creation_input_tokens': 2,
                        'cache_read_input_tokens': 3,
                        'input_tokens': 1,
                        'output_tokens': 1,
                    },
                )
            ),
            id='AnthropicMessage-cached',
        ),
        pytest.param(
            lambda: BetaRawMessageStartEvent(
                message=anth_msg(BetaUsage(input_tokens=1, output_tokens=1)), type='message_start'
            ),
            snapshot(RequestUsage(input_tokens=1, output_tokens=1, details={'input_tokens': 1, 'output_tokens': 1})),
            id='RawMessageStartEvent',
        ),
    ],
)
def test_usage(
    message_callback: Callable[[], BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent], usage: RunUsage
):
    assert _map_usage(message_callback(), 'anthropic', '', 'unknown') == usage


def test_streaming_usage():
    start = BetaRawMessageStartEvent(message=anth_msg(BetaUsage(input_tokens=1, output_tokens=1)), type='message_start')
    initial_usage = _map_usage(start, 'anthropic', '', 'unknown')
    delta = BetaRawMessageDeltaEvent(delta=Delta(), usage=BetaMessageDeltaUsage(output_tokens=5), type='message_delta')
    final_usage = _map_usage(delta, 'anthropic', '', 'unknown', existing_usage=initial_usage)
    assert final_usage == snapshot(
        RequestUsage(input_tokens=1, output_tokens=5, details={'input_tokens': 1, 'output_tokens': 5})
    )


async def test_anthropic_model_empty_message_on_history(allow_model_requests: None, anthropic_api_key: str):
    """The Anthropic API will error if you send an empty message on the history.

    Check <https://github.com/pydantic/pydantic-ai/pull/1027> for more details.
    """
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run(
        'I need a potato!',
        message_history=[
            ModelRequest(parts=[], instructions='You are a helpful assistant.', kind='request', timestamp=IsDatetime()),
            ModelResponse(parts=[TextPart(content='Hello, how can I help you?')], kind='response'),
        ],
    )
    assert result.output == snapshot("""\
I can't physically give you a potato since I'm a digital assistant. However, I can:

1. Help you find recipes that use potatoes
2. Give you tips on how to select, store, or prepare potatoes
3. Share information about different types of potatoes
4. Suggest where you might buy potatoes locally

What specific information about potatoes would be most helpful to you?\
""")


async def test_anthropic_web_search_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebSearchTool()], model_settings=settings)

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather in San Francisco today?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about the weather in San Francisco today. This is asking for current, real-time information that would require a web search since weather conditions change frequently and I need up-to-date information. According to the guidelines, I should search for current conditions or recent events, and this clearly falls under that category.

I should search for San Francisco weather today to get the most current information.\
""",
                        signature='Et0ECkYIBxgCKkCXTXBKWJ3QYffHphenTDDE5jxo/vbyyvFuY7Gi5PGLYFdjxF0KQ4BGT7bGzB53hSRPgJtjUD975U7TZ4f9IheWEgy4pMKmvEJ0D9XDrxsaDDpjMZqhX/EnpJmjGyIwreKtd2Xj+RpguF1YI50dldiwk6qQNW2rK+xLwmWY5qF75b7WZrmOZ3endXYEQjBMKsQDmsnYnUODvD5Uh/yRIUgOp+6P5JrYjLabtsC3wfuIISLVe5QhC/3Ep7K/x55u97qy/DIhCAOz38x4YId37Pqq8XARrRq5CPwzxBzsMfPwpeV5eRHLQmasZxpOhivd1lMLC7B6D9EdpWefKWE+Ux1cMxpfaQj45cpMn93qLyCLGtNqnZJ2nPT7eoOtavZ9VvN5LsJOIWYEkxK+iq/6XYSJE5JlqBtDt9Y5P1QT/QnhFwfxjD/Cs3+RrGzKp2loEjmeYzNBwEfbY+pyKHJUS3bsxWyyi0d9Gc6Zfj4Xiuf/G0ninvXpSQheXi5gcvqIir6ZhcC40vHwvdVtJipSLkqMoPQcppCTOa2ATFyLKZIlug2OjoWIHrC5xnkCuKLXVMtHTF0mdrW0R/SgecnequYprzPeCc+Niqf4CVk62qtp+H06oWKQvHbP+s7kuAbdnhJjkcETiN8fP7+eLzKjRFAVnT0tixaNFjB6lWbg2ePyQDhqeVn6i/ULCzKyoY/hSIfZXUFwTCSDW42WvITFfPfWBBW+p6R/8peJ/KS2q0wHT2G3N4N7xFaNLOTXE0iPPtWsdqZw4cNQi9IUGKayqZ+/02tJYaEYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'San Francisco weather today'},
                        tool_call_id='srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'National Weather Service',
                                'type': 'web_search_result',
                                'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                                'type': 'web_search_result',
                                'url': 'https://www.nbcbayarea.com/weather/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://abc7news.com/weather/',
                            },
                        ],
                        tool_call_id='srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, here's the weather information for San Francisco today (September 16, 2025):

**Current Conditions:**
- \
"""
                    ),
                    TextPart(
                        content='Temperature: 66°F with clear skies',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                                    title='San Francisco, CA Weather Forecast | AccuWeather',
                                ),
                                cited_text='San Francisco, CA Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · News · F...',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDNPrwrtV82S39TBtUxoM9OkESVCLqFIhGNWNIjApHhwZq58odSIAc5nQHEhQBfNwFHrpQpbkerxt3aYCRE+4AgDlX+pT8ZzljoUJKlsqE58zTWv5W9dV59ikkUJ6StlyfaMYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Wind: W at 3 mph with gusts up to 5 mph',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                                    title='San Francisco, CA Weather Forecast | AccuWeather',
                                ),
                                cited_text='San Francisco, CA Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · News · F...',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDI1rQWxpfVxj9JrdJhoMHMrF/xm5CW6KLkbfIjBeUMWBFegc/AgNXWM7fzL3+LhX2Tqt8kJgrbdgEwNARhO5VYMDMP9tC833w5Xmb6UqE0jbo7+GooCq/w5RRvX1KgE0zhkYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Air quality is poor and unhealthy for sensitive groups',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                                    title='San Francisco, CA Weather Forecast | AccuWeather',
                                ),
                                cited_text='The air has reached a high level of pollution and is unhealthy for sensitive groups. ',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDKBO3m5oU3zDP/M1lBoMBKa8Z3revdebJHWbIjCRSJ1/FdR/uZeWZy5x85sd7yfm0SW+4URT2sN/CN5Qf9fQpe/sppMjAby+dqZg6bcqE3MW5v2cyJybai3gEjOauAM3d+EYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


**Today's Forecast:**
- \
"""
                    ),
                    TextPart(
                        content='High: 78°F with partly cloudy skies',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/us/ca/san-francisco',
                                    title='San Francisco, CA Weather Conditions | Weather Underground',
                                ),
                                cited_text='TomorrowTue 09/16 High · 78 °F · 8% Precip. / 0.00 °in Partly cloudy skies. High 78F. ',
                                provider_details={
                                    'encrypted_index': 'EpEBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDMEHnsQLGWLb7FdzCRoMa7WCRfpvJ28BCN8yIjAXclnerIt70cUarlOsWiFEjbDmU36iMFVEKtYsc69NdaSgmZwGs/goHJHgmoEjpQ8qFZb8U5NbegnKXzYWdQ+V496rsOBjkRgE'
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Winds W at 10 to 20 mph',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/us/ca/san-francisco',
                                    title='San Francisco, CA Weather Conditions | Weather Underground',
                                ),
                                cited_text='Winds W at 10 to 20 mph.',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDLEc67xHbdy4qfeR/RoM9BGaTSFHONyszvURIjCCgwrv6kbCYL5PMYfS17Bmt3ftzKrhVZo9rzdzjFTnyi8RzShtAL61Zi6mCV+X0XwqE5e5yOh/SmB3xxNzuVuwqzKlyYgYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='8% chance of precipitation',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/us/ca/san-francisco',
                                    title='San Francisco, CA Weather Conditions | Weather Underground',
                                ),
                                cited_text='TomorrowTue 09/16 High · 78 °F · 8% Precip. ',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDOE7SfGqIX9sRB5ErBoMLJgiHrETP3PY6+9tIjBSx9+ZtuCyLXTzqikWSduGv7iRU5q/EtnhNAaM5IWvMq34A5XMxeCLlBhzIrEwkGcqE5pflUMu5Du+vUwg1F8Dw+ILPm4YBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Some clouds in the morning will give way to mainly sunny skies for the afternoon',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/hourly/us/ca/san-francisco',
                                    title='San Francisco, CA Hourly Weather Forecast | Weather Underground',
                                ),
                                cited_text='Some clouds in the morning will give way to mainly sunny skies for the afternoon.',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDCXpOg0iIqnZzS5IyhoMLxGJh5gs25sqXONYIjAPYG5Xp0rjytJih22sUIdxlvoJk/n46zm6bW4xdjOFtgx0Ew/lavCN/f+ZR3s27bwqFDk8+RaU6EkOC679jQMRRCzJawWsGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


**Tonight:**
- \
"""
                    ),
                    TextPart(
                        content='Low: 57°F with clear to partly cloudy conditions',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/us/ca/san-francisco',
                                    title='San Francisco, CA Weather Conditions | Weather Underground',
                                ),
                                cited_text='Radar · Satellite · WunderMap|Nexrad · TonightMon 09/15 Low · 57 °F · 8% Precip. / 0.00 °in Clear to partly cloudy. Low 57F. ',
                                provider_details={
                                    'encrypted_index': 'EpEBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDCXHqEwUsSE26Sb2gRoMuO3SVDszvhU5huc/IjCI4LbKsCWba2Lf29G9dmwVOEdChsl/Fnjc1XwuCvUSBSyf8drc88Xc9zaLSiSzIUoqFYoa+znLyYC7wsiPLuMshhZ7CGv4HhgE'
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Winds W at 10 to 20 mph',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/us/ca/san-francisco',
                                    title='San Francisco, CA Weather Conditions | Weather Underground',
                                ),
                                cited_text='Winds W at 10 to 20 mph. ',
                                provider_details={
                                    'encrypted_index': 'Eo8BCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDGo4gB0zmLZBHsIkrBoM5bL+sdYKbftyuFMvIjCe2bmPQnUmWiZHZuRPQnVdpmbr/AZd+o9HIXtVeVwLU/fvqLxlk5WTP6GqrKnU1WoqE+4HQ3DNwQ7rFDsZum6duqskmHoYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


Overall, it's a pleasant day in San Francisco with mild temperatures and mostly sunny conditions, though the air quality is poor, so sensitive individuals should limit outdoor activities.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=8984,
                    output_tokens=520,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 8984,
                        'output_tokens': 520,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0119wM5YxCLg3hwUWrxEQ9Y8',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is now asking about the weather in Mexico City today. I should search for current weather information for Mexico City.',
                        signature='EqgCCkYIBxgCKkAhyrWtc4MfwZtLCpH/f41h3xS0UBTKetW5LA6ADj/q/8G5GiD+31L8MWU5+8QbLKrdzKIr5RZTEmval6pjPCxwEgygcM1WHSKHKa3PiscaDDtaNmY6L04w/DaCFSIw4mjvUNimq2ShpHNyVrezsnnXaRyyt2Ei4Iik2sCgzARFHGyDNzerHS/aCxzMR8MFKo8BVo7IxMBObxJIn43oG4aHroTyH4tX0IB3HPE1L1O/RZ9HfrmCc/KJwvIc79klaolMdyFvc343GJbssZxF1YJ+8YgGJtrzsKaawjsNelJBqkNWdF/TFwY0G+zGS90yWmHp4hFylIib5OTYz1Dm8O066biiZps8EDkINIoiIfkslPdnP3FWiCl9g6+gSiJd+WwYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Mexico City weather today'},
                        tool_call_id='srvtoolu_01SnV7n4h3ZQtz14JriSp4xa',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 month ago',
                                'title': 'Weather Forecast and Conditions for Mexico City, Mexico - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'August 12, 2025',
                                'title': 'Weather Forecast and Conditions for Cuauhtémoc, Mexico - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/Cuauht%C3%A9moc+Mexico?canonicalCityId=7164197a006f4e553a538a0b73c06757',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, CMX, MX Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/mx/ciudad-de-mexico/mexico-city/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, Mexico 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/mx/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'August 12, 2025',
                                'title': 'Mexico City, Mexico Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/mx/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'June 19, 2025',
                                'title': 'Weather for Mexico City, Ciudad de México, Mexico',
                                'type': 'web_search_result',
                                'url': 'https://www.timeanddate.com/weather/mexico/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': '10-Day Weather Forecast for Mexico City, Mexico - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Yr - Mexico City - Hourly weather forecast',
                                'type': 'web_search_result',
                                'url': 'https://www.yr.no/en/forecast/hourly-table/2-3530597/Mexico/Mexico%20City/Mexico%20City?i=0',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': '10-Day Weather Forecast for Cuauhtémoc, Mexico - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/Cuauht%C3%A9moc+Mexico?canonicalCityId=7164197a006f4e553a538a0b73c06757',
                            },
                        ],
                        tool_call_id='srvtoolu_01SnV7n4h3ZQtz14JriSp4xa',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, here's the weather information for Mexico City today (September 16, 2025):

**Current Conditions:**
- \
"""
                    ),
                    TextPart(
                        content='Temperature: 59°F (15°C) with clouds and sun',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                                    title='Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                ),
                                cited_text='Mexico City, México City Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · N...',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDAOzK/sV6nMomFW2kRoM2cZqq0m0dYUhtJoKIjCRoWEa2lCJ9f6+ryX0trWqYsZ6rPS/y95nqKKwPL29Y3kIBYIpLxqwHyVOEZk1tqEqFIdVvN8sCIb2eBLv9bD+uASv84RIGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Wind: NNE at 6 mph with gusts up to 6 mph',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                                    title='Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                ),
                                cited_text='Mexico City, México City Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · N...',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDBwX9yHgJizGz+POZBoM92BMeHad7MAD4MwrIjABD1bm8ylowz91g3mUfcz6MYTpvwSopPi9m5ScR23vIJPnjRSdAUvZJ5rSRq3xbhMqFK/lV236LtEbTNOm/AI6IEfGR837GAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Air quality is poor and unhealthy for sensitive groups',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                                    title='Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                ),
                                cited_text='The air has reached a high level of pollution and is unhealthy for sensitive groups. Reduce time spent outside if you are feeling symptoms such as dif...',
                                provider_details={
                                    'encrypted_index': 'EpIBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDKRmKtS16X3NrfJ2QxoMUXKCzdBLTUVevFWxIjCfmrUGroDcg5sEEQ+dokyvxWjsYUx9q8t2PjAoYgJHj3J4QIhsr07e9R7aw8vyeDEqFhNpLn2HEep2d7AEtxBQUHpOUWA0znYYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


**Today's Forecast:**
- \
"""
                    ),
                    TextPart(
                        content='High: 72°F (22°C) - mostly cloudy with a touch of rain this afternoon',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                                    title='Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                ),
                                cited_text='Mexico City, México City Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · N...',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDCLnLxDJlNvAxnREgRoMXCDRJkOVPw+a9Ao8IjDL4ajzIFIvCksE+thOjKlvk3/bKLZ4xvli5rZ+SMzAedT5axFxgeAZdi6hESeYOSwqFBaW1SxkYu3Al9bG7pfeh+yPfkxKGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='High 73F with partly cloudy conditions early followed by scattered thunderstorms. Winds NNE at 10 to 15 mph, 70% chance of rain',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://weather.com/weather/tenday/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                                    title='10-Day Weather Forecast for Mexico City, Mexico - The Weather Channel | weather.com',
                                ),
                                cited_text='High 73F. Winds NNE at 10 to 15 mph. Chance of rain 70%. ',
                                provider_details={
                                    'encrypted_index': 'EpIBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDHSZ3uH4UirBmEnFABoM2eVVOYVeni424YvSIjB+qwFBjLzC2vjBViOlGx/nY7hfYV2MmH4npc7NEcj9c+GYjpv8Sq37dct3xXn5xQkqFqRBDsCci3pA3bYtDzeWJPXkGfStN7gYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Scattered thunderstorms developing during the afternoon. High near 75F with winds NNE at 10 to 15 mph and 70% chance of rain',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/mx/mexico-city',
                                    title='Mexico City, Mexico Weather Conditions | Weather Underground',
                                ),
                                cited_text='TomorrowTue 09/16 High · 75 °F · 71% Precip. / 0.11 °in Scattered thunderstorms developing during the afternoon. ',
                                provider_details={
                                    'encrypted_index': 'EpQBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDKf1qnGy8I0eR8ESsBoM4aCgn1wCp8bYcaIiIjAIU5E2GBPS6tbnFzEvPUc8GKEzHQpu1LDJTmRClIalH6TgGQl1kX9Al5KeLW3rp84qGDzo/SAht1QADq0c8i3j1EYhbIIHs48DchgE'
                                },
                            ),
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.wunderground.com/weather/mx/mexico-city',
                                    title='Mexico City, Mexico Weather Conditions | Weather Underground',
                                ),
                                cited_text='Winds NNE at 10 to 15 mph. Chance of rain 70%. ',
                                provider_details={
                                    'encrypted_index': 'EpQBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDAVUj8aNUG2A8w99+BoMywDl5V+elFPoeYQ0IjBcnmKXAiKBffdxM4tAD4Dc1VZBn3RY35CDywsdaQcH+otKGMdHAZxQ8bOOCu5IUV0qGEqLDqR9Mj84lqvDbdSMyWwHxoM0fpTYERgE'
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


**Tonight:**
- \
"""
                    ),
                    TextPart(
                        content='Low: 58°F with cloudy conditions and a couple of showers',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                                    title='Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                ),
                                cited_text='Mexico City, México City Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · N...',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDFdGuXvbuEQvMD5x1RoMa3bhIXG3ScZpYg12IjCg2r64JNEa8Pdye/LW8zSZhQIxCAatSa4pnbUTVEPhtP5s2XNqUOgvtN9jwoJiFcwqFFfVb7qKSu5o+XuprTlYyQQzbJsZGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Cloudy overnight with low 57F and winds NNW at 10 to 15 mph',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://weather.com/weather/tenday/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                                    title='10-Day Weather Forecast for Mexico City, Mexico - The Weather Channel | weather.com',
                                ),
                                cited_text='Heads-up · Gabrielle Likely, Plus Second Area To Watch · Humidity58% UV IndexExtreme · Sunrise6:24 am · Sunset6:37 pm · 57° · 24% NNW 11 mph · Cloudy....',
                                provider_details={
                                    'encrypted_index': 'EpIBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDAc8ulmazXGe892FkRoMa5fdGk7fmU3xSOLUIjDtOelcSTGskW6wL7uU2ptERJVnZK7Q1VVC6Yl/M2k2FFVU0DjruHMaIC+AI2PYs0AqFqqaZ/5Wiwcb2VSxrzpaLEF+yiCO3aIYBA=='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


Mexico City is experiencing typical rainy season weather with moderate temperatures, high humidity, and afternoon thunderstorms expected. Like San Francisco, the air quality is poor, so those with respiratory sensitivities should take precautions.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=19859,
                    output_tokens=544,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 19859,
                        'output_tokens': 544,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Vatv9GeGaeqVHfSGhkU7mo',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_web_search_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebSearchTool()], model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about the weather in San Francisco today. This is clearly a request for current, real-time information that changes daily, so I should use web search to get up-to-date weather information. According to the guidelines, today's date is September 16, 2025.

I should search for current weather in San Francisco. I'll include "today" in the search query to get the most current information.\
""",
                        signature='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args='{"query": "San Francisco weather today"}',
                        tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'National Weather Service',
                                'type': 'web_search_result',
                                'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                                'type': 'web_search_result',
                                'url': 'https://www.nbcbayarea.com/weather/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://abc7news.com/weather/',
                            },
                        ],
                        tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Based on the search results, I can see that the information is a bit dated (most results are from about 6 days to a week ago), but I can provide you with the available weather information for San Francisco. Let me search for more current information.'
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args='{"query": "San Francisco weather September 16 2025"}',
                        tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco weather in September 2025 | Weather25.com',
                                'type': 'web_search_result',
                                'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                                'type': 'web_search_result',
                                'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                                'type': 'web_search_result',
                                'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco weather in September 2025 | California',
                                'type': 'web_search_result',
                                'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, Weather for September, USA',
                                'type': 'web_search_result',
                                'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '3 weeks ago',
                                'title': 'September 2025 Weather - San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco Weather in September | Thomas Cook',
                                'type': 'web_search_result',
                                'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '4 days ago',
                                'title': IsStr(),
                                'type': 'web_search_result',
                                'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                            },
                        ],
                        tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, I can provide you with information about San Francisco's weather today (September 16, 2025):

According to AccuWeather's forecast, \
"""
                    ),
                    TextPart(
                        content='today (September 16) shows a high of 76°F and low of 59°F',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                                    title='San Francisco, CA Monthly Weather | AccuWeather',
                                ),
                                cited_text='San Francisco, CA Weather Today WinterCast Local {stormName} Tracker Hourly Daily Radar MinuteCast® Monthly Air Quality Health & Activities · News · F...',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDF40DRRSNnVV3UQhMhoME3fVNuLj0UFoYnjZIjCgodwR9nJs5Ax0wYOB3qnP6aRoPSxxmbCs88auNlqqTGUqZRWBAI4N0By5bOAFJF4qFK0WEkiPPIMDb4tVJTRA9sSl86GQGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\
 for San Francisco.

From the recent San Francisco Chronicle weather report, \
"""
                    ),
                    TextPart(
                        content='average mid-September highs in San Francisco are around 70 degrees',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                                    title='Here’s when S.F. weather could hit 90 degrees next week',
                                ),
                                cited_text='Average mid-September highs in San Francisco are around 70 degrees. ',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDH3EgputwhrYmTSWkBoMR1g/HqJaXbK1bsBQIjCWmKqwNmnpvSdJS5cG8TKXe2SbNjMAZO8j7W7YteGkOGvLBxnrcMOTh9TjRvXjm28qFPP2EEUPRVdFRg/khJM0JcBYs/ceGAQ='
                                },
                            ),
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                                    title='Here’s when S.F. weather could hit 90 degrees next week',
                                ),
                                cited_text='Average mid-September highs in San Francisco are around 70 degrees. ',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDFDx+f2Hj6wkNpswqhoMCeUDlHiMe0i9t1e6IjBRNMm9hMzr3lAuli5aRzDN/kyG28rjZyS0CUffCkSklmI0hunp67hVGP/tbzeZgYUqFGKgbS6TfBOX/a6hKPiL14JUl/udGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\
, so today's forecast of 76°F is slightly above the typical temperature for this time of year.

The general weather pattern for San Francisco in September includes:
- \
"""
                    ),
                    TextPart(
                        content='Daytime temperatures usually reach 22°C (72°F) in San Francisco in September, falling to 13°C (55°F) at night',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.weather2travel.com/california/san-francisco/september/',
                                    title='San Francisco weather in September 2025 | California',
                                ),
                                cited_text='... Daytime temperatures usually reach 22°C in San Francisco in September, falling to 13°C at night. ',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDJbvWTEObIENPeg2jxoMXGlxzP1Uyuc+2hsWIjAg8u0sUvYEYFdEYiwBb1GkAx1RkamSZ0KreMTOu5X4ZNGOYwWkVPANKEnI7gqrSlcqFPffLYpNc62cyXWiS6+fTmf3sSo5GAQ='
                                },
                            ),
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.weather2travel.com/california/san-francisco/september/',
                                    title='San Francisco weather in September 2025 | California',
                                ),
                                cited_text='... Daytime temperatures usually reach 22°C in San Francisco in September, falling to 13°C at night. ',
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDHzLg5l+tM/ffc+sIxoMu0XLITxRs9iGUiOZIjCVw6Spvws/dkAngK7JhGYyexOac5AFH1ay+7P5X7uzdQ45HenkLIIQGQFGTbWtxREqFKwjsGwPjTRTli9ipIJTUqNLTKAZGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='There are normally 9 hours of bright sunshine each day in San Francisco in September',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://www.weather2travel.com/california/san-francisco/september/',
                                    title='San Francisco weather in September 2025 | California',
                                ),
                                cited_text="There are normally 9 hours of bright sunshine each day in San Francisco in September - that's 73% of daylight hours. ",
                                provider_details={
                                    'encrypted_index': 'EpABCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDLsYD2oWQxpWeTy+hRoMIGR+pZTWCOBtK8vuIjDPGODqF/O0ikxRjnXzksLRdiadJh+z0bD0mNgSUQkpukodMk8Mj7njCWAzjKIvtSIqFDBrpyn6zsCYYFCGRmOozHbz/hmLGAQ='
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm. Typically, there are no rainy days during this month',
                        citations=(
                            Citation(
                                source=UrlCitationSource(
                                    url='https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                                    title='Weather San Francisco in September 2025: Temperature & Climate',
                                ),
                                cited_text='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm. Typically, there are no rainy days during this mon...',
                                provider_details={
                                    'encrypted_index': 'EpQBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDMgXE4UZcyDxqPN/kxoMMZjABtflMw7yLqFFIjDsML/HQ1FAol5ql8svI/FEaOrOVDUaus1aZb492kQtvkH86qCbMJUTQ73GGcmRQz8qGIZij1uOCYKSArWfTnyncHgnP0CtoNGo0RgE'
                                },
                            ),
                        ),
                    ),
                    TextPart(
                        content="""\


So for today, you can expect partly sunny to sunny skies with a high around 76°F (24°C) and a low around 59°F (15°C), with very little chance of rain. It's shaping up to be a pleasant day in San Francisco!\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=22397,
                    output_tokens=637,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 22397,
                        'output_tokens': 637,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01QmxBSdEbD9ZeBWDVgFDoQ5',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='The user is asking about the weather'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' in San Francisco today. This is clearly a request'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' for current, real-time information'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' that changes daily, so I should use'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' web search to get up-to-date weather'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' information. According to the guidelines, today'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s date is September 16, ")),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
2025.

I should search for current\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' weather in San Francisco. I\'ll include "'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='today" in the search query to get the most current'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' information.')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ=='
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
The user is asking about the weather in San Francisco today. This is clearly a request for current, real-time information that changes daily, so I should use web search to get up-to-date weather information. According to the guidelines, today's date is September 16, 2025.

I should search for current weather in San Francisco. I'll include "today" in the search query to get the most current information.\
""",
                    signature='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ==',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h', provider_name='anthropic'
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"query": ', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='"Sa', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='n Fr', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='anc', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='isc', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='o weather', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=' tod', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='ay"}', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather today"}',
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'National Weather Service',
                            'type': 'web_search_result',
                            'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcbayarea.com/weather/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Current Weather - The Weather Network',
                            'type': 'web_search_result',
                            'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://abc7news.com/weather/',
                        },
                    ],
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=3, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d on the search results, I can see')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' that the information is a bit date')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d (most results are from about 6')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' days to a week ago), but I can provide')),
            PartDeltaEvent(
                index=3,
                delta=TextPartDelta(content_delta=' you with the available weather information for San Francisco.'),
            ),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Let me search for more current')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' information.')),
            PartEndEvent(
                index=3,
                part=TextPart(
                    content='Based on the search results, I can see that the information is a bit dated (most results are from about 6 days to a week ago), but I can provide you with the available weather information for San Francisco. Let me search for more current information.'
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx', provider_name='anthropic'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='quer', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='y": ', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='"San', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta=' Fra', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='nci', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='sco w', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4,
                delta=ToolCallPartDelta(args_delta='eather S', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx'),
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='ep', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4,
                delta=ToolCallPartDelta(args_delta='tember 16 2', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx'),
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='025"}', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartEndEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather September 16 2025"}',
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | Weather25.com',
                            'type': 'web_search_result',
                            'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                            'type': 'web_search_result',
                            'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                            'type': 'web_search_result',
                            'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | California',
                            'type': 'web_search_result',
                            'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, Weather for September, USA',
                            'type': 'web_search_result',
                            'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 weeks ago',
                            'title': 'September 2025 Weather - San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco Weather in September | Thomas Cook',
                            'type': 'web_search_result',
                            'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 days ago',
                            'title': IsStr(),
                            'type': 'web_search_result',
                            'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                        },
                    ],
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=6, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(
                index=6,
                delta=TextPartDelta(
                    content_delta="d on the search results, I can provide you with information about San Francisco's weather"
                ),
            ),
            PartDeltaEvent(
                index=6,
                delta=TextPartDelta(
                    content_delta="""\
 today (September 16, 2025):

According\
"""
                ),
            ),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta=" to AccuWeather's forecast, ")),
            PartDeltaEvent(
                index=7, delta=TextPartDelta(content_delta='today (September 16) shows a high of 76°F and low of 59°F')
            ),
            PartEndEvent(
                index=6,
                part=TextPart(
                    content="""\
Based on the search results, I can provide you with information about San Francisco's weather today (September 16, 2025):

According to AccuWeather's forecast, \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=8,
                part=TextPart(
                    content="""\
 for San Francisco.

From the recent San\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=8, delta=TextPartDelta(content_delta=' Francisco Chronicle weather report, ')),
            PartDeltaEvent(
                index=9,
                delta=TextPartDelta(content_delta='average mid-September highs in San Francisco are around 70 degrees'),
            ),
            PartEndEvent(
                index=8,
                part=TextPart(
                    content="""\
 for San Francisco.

From the recent San Francisco Chronicle weather report, \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=10,
                part=TextPart(content=", so today's forecast of 76°F is"),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=10,
                delta=TextPartDelta(
                    content_delta="""\
 slightly above the typical temperature for this time of year.

The\
"""
                ),
            ),
            PartDeltaEvent(
                index=10,
                delta=TextPartDelta(
                    content_delta="""\
 general weather pattern for San Francisco in September includes:
- \
"""
                ),
            ),
            PartDeltaEvent(
                index=11,
                delta=TextPartDelta(
                    content_delta='Daytime temperatures usually reach 22°C (72°F) in San Francisco in September, falling to 13°C'
                ),
            ),
            PartDeltaEvent(
                index=11,
                delta=TextPartDelta(content_delta=' (55°F) at night'),
            ),
            PartEndEvent(
                index=10,
                part=TextPart(
                    content="""\
, so today's forecast of 76°F is slightly above the typical temperature for this time of year.

The general weather pattern for San Francisco in September includes:
- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=12,
                part=TextPart(
                    content="""\

- \
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=13,
                delta=TextPartDelta(
                    content_delta='There are normally 9 hours of bright sunshine each day in San Francisco in'
                ),
            ),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' September')),
            PartEndEvent(
                index=12,
                part=TextPart(
                    content="""\

- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=14,
                part=TextPart(
                    content="""\

- \
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=15,
                delta=TextPartDelta(
                    content_delta='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm.'
                ),
            ),
            PartDeltaEvent(index=15, delta=TextPartDelta(content_delta=' Typically, there are no rainy days')),
            PartDeltaEvent(index=15, delta=TextPartDelta(content_delta=' during this month')),
            PartEndEvent(
                index=14,
                part=TextPart(
                    content="""\

- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=16,
                part=TextPart(
                    content="""\


So for today, you can expect partly sunny to sunny skies with a\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=' high around 76°F (24°C)')),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=' and a low around 59°F (15°C),')),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=" with very little chance of rain. It's sh")),
            PartDeltaEvent(
                index=16, delta=TextPartDelta(content_delta='aping up to be a pleasant day in San Francisco!')
            ),
            PartEndEvent(
                index=16,
                part=TextPart(
                    content="""\


So for today, you can expect partly sunny to sunny skies with a high around 76°F (24°C) and a low around 59°F (15°C), with very little chance of rain. It's shaping up to be a pleasant day in San Francisco!\
"""
                ),
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather today"}',
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': 'EroTCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDGwpqkFZ9MrXRilipBoMkaFQrDK2VBha5sdoIjDvKSy6OkyPcFjuZ5KLzUqVR96F1F7CMrc/ON8ZDG1+CbRPNE/JvbWdVA6JiBvF9ZAqvRK8Q57eDwlaDNidwAd2cnmQUIBGEDvYySUWwLvemlQGAdurhHZR8W5942iwICdA/PHtIX36y4yKZOF5u2lpB0N82TkccXarRKuv8uM6TgviUiePChpTLppaX3s+kU0457sxeXhFr2bGfgPnEAi0IrowzrgBapC18xmKBjmm9gdXMWJPEd7TQBDsZeswc5SDtZRwq+FFZc2PAWwWkGb34s/ajuoyriQ64qJurFaQrHPgsqaoV4MLhFbvWnwsm+TeezWUx3c0HK8s+mBPn+qrvUC/ovEzfSCqhwxHgmT7o14wh2lNZpOwpv38GDNoEu0MYmSFwwKpZrlTKqUxf0C7wxdCWUnW9hllCTWzKJb2iguPF/UPgkLCZt9Gcxfe/H3Pk6PCYVhWD47XREAfGqooRj93LjrScuWB3NZ7D4ac0m2JPIyuxG3kAxbi9L4U+Tuu4eVVYV8jAMyXGfDPbEUPM6AEOeFHx0fN8Uj3evebglsoAniV3MS3Ql6CW+JA0B1ueHNMokWrnHbpohfwHSb4t8KIzVqVyXayfFKqH4HsN25hk7AqJUWg3Wv0whhVNYqNmUm8eBKE2NsHCZvkfI5Om3nfCBuCVM9eMtQUtKfm7JTT1SzxfwGdKIM8s1RKiEXyyON9A+9pBOdo3WY7d00Ndg29VRxZ3poyuLouURYbYlWokfJO+GVCjhegq7gRO6MQtTbrTlWToSMfTGjxCDmSxS+4puD924HWZ7T1341aBvQ2Ywb0pzbKdpAEkW/uheGvUfnACgM/S9sH+BObK8vwwJ68VlH8UBc1KLGz3WcxDYKQlFp9Gwmh2tdCnZ6qDlS47TYSL7k5n7s/US+T0nfPTdv3FQ6fXh18EQl1gmqgOqlLUb6hdxF7GOtq4cKnn+pWTBNXlgNu11hB9zGDYRhN7n3qZfRzjsfX+T4wdgzhpdNfvuyP3qhMueypPaULU0dKToGxwAvio4klOrxlyNIMgYnQYBZCvfElq7eX+CTCXTOED8OYE1kwA6pAlS4fyt8SQ7DO2P4M5kZnXjjlNn7kvCBkqt6AzOevtxlmKyggsSpwJVtmryGAVbxozEWm2soEqenzVdpoUzaX42d/kDA7RJIuIc80CposF/70oe7FbNKSBj+rAGWsu/1qi2o/mMnleJRat9dCZrjYOvJZ5gWwpusQ/rH07bWJH4OpZtpmTpGu08G1TIkipqwKD5mBRiKSZf/D6hpCM2HA4M9ypvxC8OC2Fy8RHSXuKMiD95hnVLnGvgPNfYCXOJU3pVzIH2s1G+a4AKFUxiqyvJgxiBDFevLL9gcI5hympWXYDgLIuTTxoHsC0l+YjYfz1Co8vyfToHniNkJn5sNte7fYFhwDXUgbL3L/epi8vICgv0R6HcT4F6sFF+x4pSX4U1AE3VlFZ8yGZHI4/Ohj4enGqpibS+zKl0W4Hyzx8LjYQLwI4TUTdOumK5zg9UJ+bhOoxMFy7XNZAhapXWvpiNNjMi55OiM4tuFmc3tPi1BRBd65WBiLfexj7z4XFnVDrO/RFwIPp6DUzWSjpUe2tg3/D2mHMQvn065OcFJWmXLpXPJhw4LvtZ2pJ7T3i5Wnq8piKUPsz7B6gq0uJnxMAgrikZHcTATRLRDHFPo7R1W3VprbzXVFPhbqMwEPjo/YBB8GZAVr4bu4fX+Ki75hZSdaZsGegPwlspG49yaCnKSBlAhR/XCT8Nlt/dLp766uJGOekRqWCkH9j4+aNyvY7z1g9JfFT5yZFXeFAWG6epiWkj6jjkhSNUXTXkk3EXQQXFE1EJjLMifG35DvUdsCT0tOpcWNe5qZqSRnMEsfahky+VZTBRhOyOFH7RsFYGs5PG3Khcm5Is+AFbLB2GEJLQQ1l+u0zEo+RkDNWmVipLEw+Lk+vUqU2rg5erCkf0b6/xSm+2RpLL4YqEJ8yT+TfO/O80nMNUn1d2ghpwC3MpWzyYVFzOiIxtUGu2d6EjFO/3zwI0PkMlTyyTFo0JMMa5VFd4nPbq/dfEZoruGMN2mHLAd4gp6Er8Qy99v59bK/XESqKCYcR0MRhJ6/y0EHBCRAOVzb8A/Vx8SFQvDHwyb0l5u8q3Pcbn64tQ+oDiUFgHGCVVKcHnH94/l8yy/pUSAHrV0N8/VDGFZ6APE+MWH5zpZG8OBADCMwI4Hev3HNy3dzQ6PpDwMQJN8BfcAcvlxrJqVVjbIUhWF5M5CiG2C+2ICZb1TEiTtp8GMPqY1XQsHq22PVC2bR4k+u4CQ9NBmY5cXnUYRtMeDJyWalLX97p42yAF66GiqGGEoRpQV/r++X2ukoolSwDkh1RwvkZUVWf3oP7TwFLzZ1qGKfBo+hN7YPTJIOGpeqSmdnqf17jQMWiSvyYpFphcqEUZchJwCzge/Q+HJ+ScHQ31lt56WVIoNCspyQA98ICCZIuarlKElAEaUVF6y5Cjb2+9ZXgw/l2RUpIBnIS2SJZE0p9PKVC6NsxlkpGJmNeWnnfBS+UvOL+5OrM1j1sGm2MKUn1HuvKKtkvttbIgZozaLR2CMu/+hIw24ox7HthbquDZj0u8zac+Pth9bS03kkydOZ7PHyAeOTB4zndFdMlgmmOyEURJ8ENup6njscv9hV7j2235kcQaL77phvu5xnWL1qsEOcSgRgBsWx9oAAaVij1hR0H/YZNiqX7fLqORaTThKf1lvy4AX9WfkgVima8jyu6Jx6jwOoN/vJV/Ulu8bQBWjp2kgYJxKkQcwGmhIaKCwtCyyagLHNfCjBlyF7yffkFCWHcn0z0rTo/pTutLh0HOKgj2+gOlG2hWSu0AV4zTMJxCvIcP2q0zReQ1SracIHp/j0WJU0LDgvnYmLkwLuyghIjMvy7O8yiUaFExVKKYIVeBu7Miy9T5oz+unvs/wARHfomL1d1W32ZDgpmAHg3GyP9d3DotODvUJMSWL1YSoWQbzWhOmhlB0GalzjQx1n9OJLRCGFSPGZD+l+0PnLk0xUyI5aDr59G4c1/LvVH0NiVTDQo7o8qLwBb7PrJgyerVNM9nT87zlRI471KTsdg3uP0sEt4LY7OqXmHlAU0KHZA7vTZwaOQb9iku+4A0GyTxG/icQ8jjAgOlkhdDVxPSC1xMkNyAKrCCnAhbVAAisfTDEV3mZi9AUqqL0DMD86916gHC6nGAM=',
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                        },
                        {
                            'encrypted_content': 'EqsCCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDLTOmzU6o2IFyelIDxoMpBI6ydDK/wwgkMRXIjAm+0O5fpxZFLA08gCLYc2s2emPmaUTCPPIMuecseNQJxE2fEEEHIZhaA92poJuL4cqrgEH6wmAzVTb8MFWgqH4lXYAEei4SLnGtDXaMJ4CWgKxqdNbwHqgncCLzIsUtTQx8lv3hfetXrVRbGGLUJ0RfhdSbykJTCSzI/23em8I11xKi9W+thDH7w/aujj8d0vhLSf35UeGo2+VNpwFSiuNhUOrPH6MfqYmzrT2jYYVDbjK9ndqrsXBiXTqvgILvrl/+l+13DCXkwfE0xcyCgwJ1o8SxbSGpTo3WPXAX/zCUxcYAw==',
                            'page_age': '6 days ago',
                            'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                        },
                        {
                            'encrypted_content': 'EpsCCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDPdRJgWfAROo9/vXwhoMM7M05zjidI1J3upEIjBfedfsWm2an6Wse3Thj6GAFYPOK8/kAcoU9H02bWqwf+bhkS97Iclko2uBm7woQJMqngG0x06jamHmqaJKrfwo3CuE3BUj4WeMQmtSdz0B5qwDL5NhWHpFIgcuyz5sJRDWNvlKHUNrXiPotMOvjD3/bigF2jzx2bRGj3Do7E7iv7r9Ne9JRuK2oSPF3sGR61v+8Z/WK6wHdHz69BUtjzaQ01MWz3KAcpX0zEWkUGcVJ0h+LhGmFl+VSy1O086l3WPpVypfDDB/fiYzsW5AAL0h3xgD',
                            'page_age': None,
                            'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': 'ErgBCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDP9OLadJrGp5B0DSKhoMgp/XmmKSbz9TF++pIjAkpViRLY9iQAx4lMZcA/BCyc0UUwFmzHTe3ET1bbvsSfx2KGU+IpS9oJgmK8eRRJ0qPBHXYyrILxqe1g/KhyyytqxgIOSBTQmWAn5Xc948TAnT9hmVUAx4qXcIA1iB4c5VtvWafXzAG4ju+dJSARgD',
                            'page_age': '1 week ago',
                            'title': 'National Weather Service',
                            'type': 'web_search_result',
                            'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                        },
                        {
                            'encrypted_content': 'EsAECioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDIC57jccAgUeWO+CmRoMiXwQw/UZ+ZUqJBgnIjBZQrLFpduzQDmOBDmD4zJH7XN4Xhlm+4I8k5HXbtWRSQqmRnk7qv6LHDUggBXpm/EqwwOcJ+yN3KaBJfiMu1/qiV8LEdmf6tCtpcAfAluL+fdu5KV4R3WHb1bDJ8IjgXxvajYyad8cHuzTPQkTZv5i30jSavz9bdSoZV/xf9tLfD+Wi1wmWpc1SfZPqKqO2qeqBDdK9a5SSbKcCpBIlwCpnPOP4CkjVuZ9QENvxDkH+uUwT1QDAeiM0m7QAFpHbosyC+QHejiTWC5jRPdZ90nxtqAk1A6MIkYb6xJh+K/XleAMM5Ym4rWhhu4nIx8MmOjLrbZkw+QuOKJKXqfzqnbEkBPqXOndkTJWPl7TvmU7j3eNDZMNlOdi1wjhDxPmjtVbktbPhuTfXc7WAUTB2PPETC9zWfaJrUBKe1XOfh35Xh2Bi54xV0MulfckJ6uuU8W1VEMgCYkIadiNwfUPE9PCqOh2WzX8e9A2FGyUI6zG/8XDf9vJqDL34xX98PagCMBxHrE/tRr7VCDWMyfrMrdr1JLAQMY1Ct6hwVjqGkY2fShTfFCBveMIqsQQC7kZJMhko7ZP5CzXUCfIEdSA3+sM8s9UMqNBjyalWoqeeR2axAkz/2Y5oW0kcy6vUxDAxyoS08YGIu5t4ER3vWElTIq3DWd1GhJfGAM=',
                            'page_age': '1 week ago',
                            'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcbayarea.com/weather/',
                        },
                        {
                            'encrypted_content': 'EocJCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDN80njAMFvO4fs5sFxoM10XYtICHmzbKOFHzIjAq5T+fi5l4yDykxiu5UqfFN0uTjIf9E9i3dYRlJb8bgPOiwapXvCdzZfYh9xGR8R8qigggVEEJ4G2xaOyV4YYD1SZ6FKWcKW/uALuPmDFs1Vrz78UHhPGgG12HpqX6AEdBon2VRf5DL899r/nNYfWM7o1jkkDpnUvWliv/OLGvd6QGFlSxGpdrmhn50y8SmuDyRNIBA6HSw6J6oR6BuwtIX3t1uGhY53UH9y7Rc5lRmWJ6CDaThLUPwuNw+W8KNAnIVX80yWXqwiP3hbR9J4xrkUirmxYtxpkJY+4qkifPfMaGuoeqrH2ahznFXEsew9GZ/Uw3nZvwF5PNrNzgamMme6KTb1Puo/QRvpGdKssn0ZhS7b9p1ZSdHndplapAk1pu4WtDCIM1ICyYeLlaIxGOI9tOMSIbPObj+OUisvXfkYRmTBJ7tIGNxr2KqWdJdKyBIo8+L/IFWIqPWicT/JZTe3cGVUy5YzRgJcX34fjtNwkcRuTteTk/gQjypbxfQHH+2RItQrP2anBWW/IYsS/4IMDUc7TI5os/x79PoWai/G9tPEEwdYGVmMaa6te/kUHyrWC6PVckP9o5OLdLqYlQpHLxOvMkuicX7m9dCs5HA4EAkAK00jSa7ppHvDsJzzVPBrseeCR8dSJ8+DuR4ScQUShKUbBFFq6A8LNI6ZRExyUud06sJ1wvHFzssbmPAIc5Te3Hb21juR0gQCWYdlyosTIXB53pGkbvKhJkf/2GG30cNsjVXStIqNUQdn1Vaa+M+dmaoYFMbuYrLmWZCmKochX/SB32fbcEyzO9mRvoKoVjgTPUbELKeNnJLZqgErtJlqY7R79Dhe154r0yASCTEjLNf/ZMfcwmq6alW0yJEB+ss6D2HI3yr02eVra8KP04jVcwAa0M3r+deeJE0dZ+8Bv2n2gyTlQjeYPFGCJtjfqbLXu/M1R9TqLgfSHD7qm3hMq+IOiKzuDZJRQPsQgMotcDFwxIbqRa0shoGchKkLWUxYsIzg922W7QCJLZ3ktWm0WzVmT9MC7uVhUBDQlCdIvJh83DEshiC+dzdJO5qJ3ptNIVDCuHEX7R8OMqFaJrsmNe+r/TPpws3flmOY7anQ/rE1d4ZgrG+YJIKuACrkTmvX8K3xhow8RCZyY6pt6ByHaTdMwNnqibkUp4wLw1lXej506+vkcSXkHXHAuoAERWPd64iGj/qsITKJWSpaGsc2thj+HcT6iSq1L6FM8CtTOwV8mHatww+X9MqemBodyOTlHbuB5MNKqDBXNzEok5h2ReH4lV9PzC3t+3udt40lLcVwmGSDXWS3/3Z28Bgj1ONO7PruP3mGhCeieNalfcip6BAYW796FkpeFlgdjnETmwNMm9a/01ZECZ1WTq+iJPNmbyB8H9PnGbx7wE0pRcR76vvfppJl2l3tUNklQniZVuJMlT7KRBD4OIjBgD',
                            'page_age': None,
                            'title': 'San Francisco, CA Current Weather - The Weather Network',
                            'type': 'web_search_result',
                            'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                        },
                        {
                            'encrypted_content': 'EvYECioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDJclKOc/QrTZQcScnxoMKCMLc4tGWeLQtqCSIjDl1WFSU5z9na9UJsXNuveNAbmnpHIhfZjZZO9VljfquIJpwvmTwcfPcXKcekK0kk8q+QOsr55hKGaocZhAiwWcPNav8OToBxbAZEo6WUYTJdJrTVLncg7S/1DBDFTBNHj69BvGRQ1GZywBkjGIpRTkuaMkO4WjwnkZ3kgH8O98GE49cjCcjeodZCvA3haKvRCEaXj/13AiNmwxUS7oPUKTl0Q49g3Zmzry8DdYwdh3tVOkN5l0DgvxeDvCJLe6plFmS90QcZx4qHmLWn4YqmgCzjyi+iRz4gnUE4/gXrhM9NZyAor9m5DFiYXtKtYx1hCC9LujrhTGR162Pxouuxl3OIV4DJBLHuLgFZP9SeODWMMz9kgGWEHLftRg5y5hZw6lnXTKoZrt4op6VrAidQfbPtCBXf2iN+wGN4+0dPdo5y6eG6EtwIUme/CR2tK8gSNpoXDEv7hhrWZZ3VG9ir5ZHlxJSz3WISZFEz6/3KLNQZd2Atm3wl7dPVjxyvFIxdnb2+r+aJosxV150PvnIZwQyYQF1r0DSFKunuzSTnOLx5WJglS5vmOH3SZV3nJfT7ZpvT2X92w6Esj6EjDs8JY5t4Ntiq7mq7f9WUK99NCyn+N+w2yx9OLc9pAP+rR1BUQBudVD8B9epjdK0nLj+nzk1b2sfhVbHnJ1TFvabdqk9PB0PKlil4K9brMJsWhIumfc60+ZB3B7SqmfUsl4TLqHJLgbBP9MtImoX2VhGAM=',
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': 'Eo8HCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDCAiv4YyS/RGnL8pvxoM3On/lSOioN+plaJSIjBkpdohA0+FeGQuOkjLJ9UmjxEnxIkUCIg88eToABjuo2Ssm18X+rCS0zNrbIcwxLgqkgZrTw2AWM6oMe5S5Cb7QtlfBYsgR2YZgAGLel0S4oUDcneQ8hSYjf+kxDbXRo/DyyNm4b4xIW68gZeOYMVoZsZ5q5RTaVbo0D/PC9SxRziI3KYheqznnn5MbsVRPZNp6u3+YjpPtM6l/GS/tEOIXIMaOIW7pG2UHpNMRwIfy6AncQWny8TNE8hksgurWFFHjr5MZgMDrl6co6jsQnyRTz7sW8Y8gmUh99oRofSJiiTsnUA8kwbKBDbco6y3TMOxV1T6I4dyyUtGAhhnaQi4mrnhFK3qrCxsb2BsF6NDTTkL3IyBAdcdcwkahfvh7Je1aqIKdKH/qX1NWjeNLOp/lo3ZmRf4GGEDC/oBoFsNkI0OthzujJmguXg3lB3+k6KpsYYfxE3jZz4PzZ1LEyNToyjkC8ZeM52aMIIYErBSl8XLwAo6Ai/8qdMDvY0iIeBZ7ypJd21rVQLIOZv0j2SRozZh2zNc0WFOfwJWUNBljvCoHKa9NINct4xeNOLWPtM550FaywVfg5g0EGwzs+W7hZcsqBJOvh8gp7Hu4PWt9fOXX6l/Dd7g9pPKGPSdXa4Zwo9RsRO2A7a/GvGnJUrM0h3Hlr6GBDbpQFNv5FSipmVLv3+OQU88wvZl5juIcHzdslu/RdRE7AA4vIhKHqqAVqvvl4JXd4d1M9fYhICCaql88qnsOPyVPZCDNAO0o/AoIi9CvneE/WT4IDoenyjCmhGCIsEXdUGYzI1WFtMEBs93224M2/obIBUQZ6VB12r2ME5yPhpAU1LIVB8Xd2Ahuy/1+yPBQwYY9eMV/p8F3w42lO9SQuP0JPy6Wvxs8LFq9lQBD7mm4ClnmdaK7jTlyVS8bMwFhQI1O7tqwF7WtULxRxmSy1NLYpJZokebulzDLq38+/ucO+pqQwPo/7d6KgObgX6l2YYAjl/glgOwyOJ3EabFNXWgIK5zq4kWFK4cq9DoqgPzfDkbbL8iE7mD3YjNqs/n6sWYn9de9kPt/5dO0M4oq7y8RMRyNbUWTUlY6M6MjDF0tCPrBqKZmCvV4wx76OgYAw==',
                            'page_age': None,
                            'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '1 week ago',
                            'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://abc7news.com/weather/',
                        },
                    ],
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather September 16 2025"}',
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': 'Eq4DCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDDPwU0Py1Q9iYNDDdhoM2qNBTnxjtd4cRn9VIjC/dZ8Ce/L8p7E9+XDGXxJOo08FcIolDYKMESXmzZ4qpTujOuYI3WINvKibmIIvpgMqsQLSU9R+vb7BJhe7mgv03VKaUqcSqu1/GtEQ2guIaydusW+HcIvamqUX9f69j/g31g7qJyK2JD26s3KEdFAqzjS7ttSXF0s5BAgIkBXn5eV/FFMxYGMiY0t3EDBM/9OOk7Of9I5RAbxYYPslJKsAovc8IuMIA/eLjYyM2wCpna+2g3VmPexJlJVJTdJPGH0Qup8HVHK3ZX563qPYXOJ2/D9ClF++e9h98EXBehmZlP/EU3zNmvX1P5XySAfILOJjiGszXjbER0E+cyY6qmJnzo8NgazPJXh+byIwsPn/UN6eHFWcA1YRtpvMzrx9Hr+AZSNhce4/Kwpwx65H5jJdY5UXDpIDXKoQf3PyaSSN9M3mcq07YgALVHBnHkldU06/472LRs+3psQaIGAwzWgeLWUW6xgD',
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | Weather25.com',
                            'type': 'web_search_result',
                            'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                        },
                        {
                            'encrypted_content': 'Eo8ECioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDNvtL6qJi8oFPXxieBoMJyFMKhdDLlHV9SCoIjC2y7zYuNlMHte8L+NnfsfrRyVmxGxNGTkwavA8xhyVYv8iFQUW8InuIa96o1QboVwqkgNQfYn+/Grceq+O1uBFfrfVP3GI86X2OXNYFx+OWWPmnWO+koYYI0UwiU7tF4BM2DWqL1uBMVabqiekDi04iKbVgrxDieEctEebaeY+cMa3fGtxlN8VyGdfKqfcHpBVt+VM81ShI/wSrc1BBtjuVRSGE9F57Ddv2T1WCfSqP8kYpV47hggTgpIPE0tJb84AIhy8jpYoRFA2JLs+MmRMVKbxLJcMrX/G5VAXmQ3HIo1dXN8JbR2JrCLGNfBmM3Yxb/HeGqGpAa1rrpKnTGOKEOTXSQYh9XTzxUBsR0SwvG7myGFwv5u8SDeaKwrmxIBe5+81o1jlCJacHXYW9ryilYavV9l8WqXuGqdty4IKU4hIKRn2v7Cm2wzr3Gt97XUcsV9RMiXEz1a6yNTl9vvhd3EzD7ivSpxxhUO6AdW3rreH0PflUub3Nt62Y8vb3HdqeWQnVzVWXD6dd3B7RuaTxC9LOiFPTw9HAeiw5lrImVPaa8FDOpP3wTG/1OybTpvAX3LirmuJMLBh21cltxZ+HDn0mmwYAw==',
                            'page_age': None,
                            'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                            'type': 'web_search_result',
                            'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                            'type': 'web_search_result',
                            'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | California',
                            'type': 'web_search_result',
                            'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': 'EqUKCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDE+3Ec8CLk6MLjz5fxoMlOJu5maBtSxHmxWLIjAQZ4b84syNFqEdmtSPXJbEeF5Ys1P3KeNJI0ZmZCgluBBTWXJtO5r0w2XLH3p5VJcqqAl6QkMfWWb+g4Qo9OetCUKXoD5UXG0/qv4eochXMgwITxRcGKgKdflIr53B1dXtCZAydP73o5MC9YajdqO73XvbtbSUespTAZjH8jHMGUGVZ1wgVG/NWY4NEO1pSdNMkGBly1dAW7QBFR3RPIyoMnfQRT86nDaJrPasdrupMlRgvcDdy+AgHZiCXMNSl1BH057q2HEnAFQHBRuKZ8Xpq5Z7ykHzbvGcXl54ciq4xev2yLmiWHNeRb3fbln1PSV4MND1dElUv0bR7DGjruUcrxAfB6Bbln4ntmYh3QFRq0azmYmFDimbVE454LANmmLtEiSQ1EgTUJIjoAKbtBB4Y3cko3S3pWs0tfyE5IonmXbQryozu433Bwy02KrJKbJ4ldHf20jzNN9FxXzXLQJhM+F91sUkeiKiNw+hSefErMSU4S4fFf4+f5ybcAGpuADxJQpr9pWdeypLGCz52trX1u2IHImg8BlUqCWgcuy6v2cFm470dMIi0M74YGi1T29NlP8mFuzthChvcRk2g/9HfwsRwNOCwy/Zn6De7MLZ/K+tnnxZMDhaZ19iWoNzf0kDcliKbdsLT82waauY37YfBaFKdbNdZBFnNGt+1bHO0db8s2AYcq/Yh4e5e1wMXrEcAnIY/IfQD2bePnqLwSAxXZ/1+4A4wQ3sDrZ/pWIKeGLfCgVw5y1MS5O/J6BZLJPcGNjtKI/FStTUGy9i/UOz9GN6jkJVzKErtZmAihbLjErnpFXBWCSlFKBGJqp+4Ob/GN0J4La7E5dA1Lx8PJHiHgUT1A3KmWk4om941EoDdf728U3L9bMSUgZSDgMoYOmLaVEhEFyu+zQZ2dC3dC9Uz5w8ltH67y7fILZzRbHxByXWeea/ObVrIqEzVjUD92/dHerMvadvZzWtmWRy+VGJETVNnxSfd2sXoRqJC+rG9o8e8YOw2o+NYeOyWInFUyGEr37zWGH+7jjMlyKIZztg/QEUmiP67zMD+ESfY7o06DEoJkyPXv/YJ3/U1vIE2n57UFSeU6tSJDgQyTZ0tgeb4ShhpEykyH8yWLFkbOyxim+Zqy0gtwjNTBXB2VqSCuXo9TwUp/RmNqE/6imUBg+ge91JZoJy1rbx3EZVHsorwHP+IQcSq4C4AMfeIFXXKHOMVUsEN9H5wt0a20SjLC/A4kM5eLQYKnpiU5bN/4OoK4VkolnanEnIywkX96Ni+zP2DQktxiKeeG4rZAYLYxF4Zei0LJhMmgD7gM8M6aH8dwMTCZME+sh2E1xoHdEa82SVQFP3Qmufk21jlFNFVzquDq9LeOY9/34yYABBraTKqtJZwf0JG/q0thvKUrLsaIJzLyz7kDuK5rMqJQ2hMJ7hP/UeQzHqXHviOMToA0PkBQ8nlPpE2Boyt2UWSI6LQKpeH/KMeYg0DmXlZ4cUv3r1gi1VxbYk7ElzhvrJKdkAU+FD1Rl+l1dL+2oQPhBQHeSQ23X/XncyXAoJNgzpnr3nEl9k0Dp4IIB9+P2B/vK2Qw3JfJyEl87ZTVfc+/DEduBOXBFEWiXZgwJ8qe6OgehBdvP5MWLM6HWcK0bgtlyR69uiDbwanNlYI2u4GAM=',
                            'page_age': None,
                            'title': 'San Francisco, Weather for September, USA',
                            'type': 'web_search_result',
                            'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                        },
                        {
                            'encrypted_content': 'Eq4CCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDB6I5l6MjKm/aCsr3hoM3n+M/m30nyfMY84mIjCOP9C1UyILIixxasmAh/zltTWIVfTkytzb85vshiTzhDGObYcEkNYBOQAuuRoj3WUqsQGTZn1oat3g8FhrHraTeG6Y4nb6j6GRZpnxKBCW9BjH2HiCR0flF8dYX39xqu088M24d03PSDGijE5j1U9IM0Fn6OjypOqUnBy1T8SfvRFTge5XGJ0bCKT/4qIEYVWA+GyrvtJXVMakQIDzD9rlA6F5j+PTqBz/Yu6joq+Zv17zvpXv9HRvrhwZdG0caTFvWPXIiHKHmxiqXg9yu0pJr6zsIrNcqpnAfNP26v+G3dGBMFoYAw==',
                            'page_age': None,
                            'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                        },
                        {
                            'encrypted_content': 'EtELCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDGpIuvOc/2zW1bt9AhoMSZxepELZpi9PBadZIjBd7J2Mr8lh0oo4ExNG292oZNyK6cYxZ91tp3ytI5vH6314M3twVG+ktzO0UUmXMMMq1Aq69HOggxg913hY5e9VfnWw6U0naNq/RJOfk6JbRygMDfBB1Ifip6bf7prOgdq/eZkrjfFiPbZAEMxxQqvm5kBj6yHZ5Vh2wo3GcGW5vEeOarlB7Y56+D3RJQB1RGhUASRGKB/nuFVkBPFm6Zt8kJoacoFzlmG4IIMzrRyfjq5WmdJ5ACjnRaikxn/glfz5PTK7hNY0a62pI31RLgOfAeYL1OKleiEsa9UBDRf/AsJVp6UGoMxQrbOhlAtlhTJvOUz+uifCcYtAo6BHvaEP7FqgASp5lMzh0bcbyEhCV/xxKHmmM0f38wMbvmjJ4JoxqNohvKO603RQ94qPkSKs585viz4+8jOel6AjS/spGGRHApNurKgFZO+WyPuQCXIW5uc3fFcYicYNwBhWiY29Z8g7mooR7siJKnNmr+D4//16WJjzz9y3ZjCBlQ/A7zCl8WnAKckXHWcZtse+ff22Qr4wk6plyGRv6RLQay92ceZnmSETzXm0jfxFD4Go2EaKapt8CaZFvpPvoZTD6iUFJlTqbSfupZfw7pY61ckwfOnIppNI27q8cwtn1Q2XwyJ1ydmISNpp2HbGG2bVtA1B/biHgjImcmygv/CKOBWdPg/e7s4BrKl72gayEZIt4g11/cBj9245HoqeCckoQhQNdBThA5milDg6mkKfQ7LlPzTJp3B294GY/rCU9X8Njzd3Chrt7h2ShxflXBWP8s4RKZYdfPNMx66lSvMZO2OyjcIv9JYKhxPqmWZeAT8BFDa/1vtsnmGS6LoVi1+NCQ/0EgUUB6Ma4QYBbmAPgHp4zoGfp3AjTmusvNyEKJUnGAt1q1m+NF/pkMbS9UwGnNz8kQvooyfRGIX/mP4yYwsKpwDHhyxFsUMUur3Tln7wWAJNRsj45WjSStwTPH7/BHYcYFnAj+T0YHd9hJvwrV+0dBUcQ3DFiws7ZTG845sP50pIALZLYh5ylw4lr9WF58CUMISUuYTQ9CClAiLr2iogI8tkI35k/FE0UuOl+ahy8a/LbO32BStJreIWPZxx8194W5Nl7u7Xpiui1odieIgLSPycMXokdys9CqFdnzvootk/J92Ms5ZfBXEPAd24H0YFU+z5bv7pBRWVx+upIwGQuxZWKAmdijmCHaZB40aD+wGZ762Gf6+n6DM1bHCdHcrZo2cRO0efAFSQGAFyEtzIlLpAaJ9+K8ryvJLDrj6MF18O4DqeHdyenQNfZk6P1i/Qce9gyeLyBE0G0jtqQ10RqAFWi9vF7CWiIvvsVu8cCz5apznJHj83BHCAUnF+GNPx8TCPAaP/YhCZW1DpOWnG1PqpT+4cPQKBFWOceCdjHExeT6YQ2AdE07tttvbTj/lna4zRNkN1iWwCn2yVx7UTXW7wO94WXXg5NDqcusBV8f/3I726QoMAf91UP8Mmb0nHPR4iPLWIcsVGwvapXmro0vCyc8tf6/JhyPguUnmXnvSydK6I1dIp9fn4W6NtF+HPfg2dtdPhjpRsElJHV3JyBQSDoSl4XlTX1s5hEAfrMJyPac6gUD8L5Ho5JGfUPOP5Xc39P4O8BA/MQIyLQjNMw2WJJ6e0oDlbinVzoXitHQpRBWsfyuTDibTZuOY1kcKYQKw3YxxLVJnEnxkdJ9kDGPbdV0iJsbJegEjFW/kUedyPcxC8TcGuFUU5ifn8/4nQdBC4B/v3/C5+nkwf/3QgVNPlAsMzFSDGcV0viaVBPLhlD2c8kgXZCLCgdK39ZvsYWFROzwldDb1yQjQckV/iPCZ1IiW5rzcXxb+2fvaPWPqRvuPlS81WdPjgZj52Ir1QK8tWzGOlfBgD',
                            'page_age': '3 weeks ago',
                            'title': 'September 2025 Weather - San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                        },
                        {
                            'encrypted_content': 'EpQKCioIBxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDOzsJ63XiM4kdNQKKhoMms7RjD9KsfN4+DYCIjBXx/iuCtyufbha9IZZqlBC5L8E3bf/dc4kwCjI6cYVptGJoNvghh8Taz38xbenGxEqlwml9hrrHT/E8A5yecu42wEaPY2k4U5JFRKAagmna9+Zx96FgTZCyixjzc9rqX3yFDmAE/6kOZrXVA3Bd322tlHf77xcRksgtkW4tq8c9N+u4io6oF0O6TC1ROhkuvwR8gHHZH6yfSPUA2adqO0a2mk8tKd6MTW259BkrGoMDo4AiZIQbeR+4UJBMW7j1/HWZSJYym7Jge/snL0e/4y7WZSi9u9O5mdeGjt1NnC/Wardgjml43Z/B2ovYtsYekKtoYsG8LorZxhpZYJ/Flk8EOnt6zBFO6AoCkruWm8QA2aBmsF7TUwweUf59zXz3NIa6WF72Rdub3TZKI26tv+Oo36Wb6CBEoOY3y5RgppayvADwPh7PTpBj6JWvij+yjNEbQK75K45scmnjkJMzfGCuNpOVJCGAvQG6Yd9DwBdCdWGSVz20j6+9xkJK142rbiHKjLabb3NGFC/WsEpWA7GdpoKMN0mDkb1p8A6V0tjfkbvAswBcY0OrLBwd96aafNGBQHI3CNbNF1VU+kbm47A6DcQj9OB0kN7nPsz/GqpTGoLamFqVzYzwlUxuPVMhV1kAino9Po0log7xSyciDoTOiiMgPED3OqJpMNy/Q6++nMkrlUlvGQduiKF+WCLrdqPcQSTVGGXfC4fcX2DVtrPNiRCn9UhrntBkjmX9nKcCcIAndEsRaeMVBDLPa0NIpbBeldl7v3Hj3LVZonpDChTRn1sLZftfGhhc0nSN5pS6k4cM1YRwPGvbBIbiWcpDMlWM9F66MiAVE0eFr/8vvtRW2Bkr+4xmm6W/iivvMdlFCBJwdeIDMaENrg8Brqmc3RH0kenYn2GPviSRAjZM2+XnQE5GZ0qfeEgLgDDipdd25ojbCGuyb6Ox/UHIoCOkyrHCa8vb3oxefaYzk0fYUhP9QSN4IDojG/INx7HP0EeIeNJCIj1UqKeffWCFamnxZYQ/2JmAH27cMu5WVmbAmo66lzMTLvSy3jOS4kYJBjyf98aHzIuAI4rriBguB4zunvnJKl+js+/rqrt10Su5H21wt4QCJbenMl0Df9TYu4zYb7WypO5GbgxWYQUjPfZ489wi9ZiBAlr+kzULa7SRWVDrLMnKjemtE3v2yRY+XStLHLWoUB9YBu7hP8RbtA1sA3gr4D5qMfulYkhCT0Mj4DBAhfrTAkoY4UPQJz8xexo87bCAsIZlu7Ahk933CE+ml5o1Ml9ujgrmEvxKQgoiUopeQ4jR0Rizsqqnhbxvu0KuzdvlZkGIJxPgPNyoV1C1SoAxUHz2oDH9oGHo5ZFhRTaKgiE5SJPy7O0H2DSznlnAUHyTCHtFAvqVXpO3OnO2zX1QAItwUd/IksDbO0XKXhcTRpbhTwErz3qJNlxrUUwCZ+2/3cFnVfJ6xUKEBvvuacQ/pmf5l35izltd2m50z/h6wjBfpuLH7cdQeSmTlBYQMfYsbmpL0CzS+3cpom0AvQbrKUdYmFudGmAiwdIOtfcziVeppyOhajYAR3gGANdBIaOZGm/EVE69VXSTfmwJj/24s0+FDYjXMC5e285TzB90MsW9Lsun6yQ7xgD',
                            'page_age': None,
                            'title': 'San Francisco Weather in September | Thomas Cook',
                            'type': 'web_search_result',
                            'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '4 days ago',
                            'title': 'Here’s when S.F. weather could hit 90 degrees next week',
                            'type': 'web_search_result',
                            'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                        },
                    ],
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_web_fetch_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebFetchTool()], model_settings=settings)

    result = await agent.run(
        'What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    )

    assert result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to fetch the content from https://ai.pydantic.dev and return only the first sentence on that page. I need to use the web_fetch tool to get the content from this URL, then identify the first sentence and return only that sentence.

Let me fetch the page first.\
""",
                        signature='EsIDCkYICRgCKkAKi/j4a8lGN12CjyS27ZXcPkXHGyTbn1vJENJz+AjinyTnsrynMEhidWT5IMNAs0TDgwSwPLNmgq4MsPkVekB8EgxetaK+Nhg8wUdhTEAaDMukODgr3JaYHZwVEiIwgKBckFLJ/C7wCD9oGCIECbqpaeEuWQ8BH3Hev6wpuc+66Wu7AJM1jGH60BpsUovnKqkCrHNq6b1SDT41cm2w7cyxZggrX6crzYh0fAkZ+VC6FBjy6mJikZtX6reKD+064KZ4F1oe4Qd40EBp/wHvD7oPV/fhGut1fzwl48ZgB8uzJb3tHr9MBjs4PVTsvKstpHKpOo6NLvCknQJ/0730OTENp/JOR6h6RUl6kMl5OrHTvsDEYpselUBPtLikm9p4t+d8CxqGm/B1kg1wN3FGJK31PD3veYIOO4hBirFPXWd+AiB1rZP++2QjToZ9lD2xqP/Q3vWEU+/Ryp6uzaRFWPVQkIr+mzpIaJsYuKDiyduxF4LD/hdMTV7IVDtconeQIPQJRhuO6nICBEuqb0uIotPDnCU6iI2l9OyEeKJM0RS6/NTNG8DZnvyVJ8gGKbtZKSHK6KKsdH0f7d+DGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'url': 'https://ai.pydantic.dev'},
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7262,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7262,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Second run to test message replay (multi-turn conversation)
    result2 = await agent.run(
        'Based on the page you just fetched, what framework does it mention?',
        message_history=result.all_messages(),
    )

    assert 'Pydantic AI' in result2.output or 'pydantic' in result2.output.lower()
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to fetch the content from https://ai.pydantic.dev and return only the first sentence on that page. I need to use the web_fetch tool to get the content from this URL, then identify the first sentence and return only that sentence.

Let me fetch the page first.\
""",
                        signature='EsIDCkYICRgCKkAKi/j4a8lGN12CjyS27ZXcPkXHGyTbn1vJENJz+AjinyTnsrynMEhidWT5IMNAs0TDgwSwPLNmgq4MsPkVekB8EgxetaK+Nhg8wUdhTEAaDMukODgr3JaYHZwVEiIwgKBckFLJ/C7wCD9oGCIECbqpaeEuWQ8BH3Hev6wpuc+66Wu7AJM1jGH60BpsUovnKqkCrHNq6b1SDT41cm2w7cyxZggrX6crzYh0fAkZ+VC6FBjy6mJikZtX6reKD+064KZ4F1oe4Qd40EBp/wHvD7oPV/fhGut1fzwl48ZgB8uzJb3tHr9MBjs4PVTsvKstpHKpOo6NLvCknQJ/0730OTENp/JOR6h6RUl6kMl5OrHTvsDEYpselUBPtLikm9p4t+d8CxqGm/B1kg1wN3FGJK31PD3veYIOO4hBirFPXWd+AiB1rZP++2QjToZ9lD2xqP/Q3vWEU+/Ryp6uzaRFWPVQkIr+mzpIaJsYuKDiyduxF4LD/hdMTV7IVDtconeQIPQJRhuO6nICBEuqb0uIotPDnCU6iI2l9OyEeKJM0RS6/NTNG8DZnvyVJ8gGKbtZKSHK6KKsdH0f7d+DGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'url': 'https://ai.pydantic.dev'},
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7262,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7262,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Based on the page you just fetched, what framework does it mention?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about what framework is mentioned on the Pydantic AI page that I just fetched. Looking at the content, I can see several frameworks mentioned:

1. Pydantic AI itself - described as "a Python agent framework"
2. FastAPI - mentioned as having "revolutionized web development by offering an innovative and ergonomic design"
3. Various other frameworks/libraries mentioned like LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor
4. Pydantic Validation is mentioned as being used by many frameworks
5. OpenTelemetry is mentioned in relation to observability

But the most prominently featured framework that seems to be the main comparison point is FastAPI, as the page talks about bringing "that FastAPI feeling to GenAI app and agent development."\
""",
                        signature='ErIHCkYICRgCKkDZrwipmaxoEat4WffzPSjVzIuSQWM2sHE6FLC2wt5S2qiJN2MQh//EImuLE9I2ssZjTMxGXZV+esnf5ipnzbvnEgxfcXs2ax8vnLdroxMaDCpqvdPKpCP3Qi0txCIw55NdOjY30P3/yRL9RF8sPGioyitlzkhSpf+PuC3YXwz4N0hoy8zVY1MHecwc60vcKpkGxtZsfqmAuJwjeGRr/Ugxcxd69+0X/Y9pojMiklNHq9otW+ehDX0rR0EzfdN/2jNOs3bOrzfy9jmvYE5FU2c5e0JpMP3LH0LrFvZYkSh7RkbhYuHvrOqohlE3BhpflrszowmiozUk+aG4wSqx5Dtxo9W7jfeU4wduy6OyEFdIqdYdTMR8VVf9Qnd5bLX4rY09xcGQc4JcX2mFjdSR2WgEJM7p5lytlN5unH3selWBVPbCj7ogU8DbT9zhY3zkDW1dMt2vNbWNaY4gVrLwi42qBJvjC5eJTADckvXAt+MCT9AAe1kmH9NlsgBnRy13O4lhXv9SPNDfk2tU5Tdco4h/I/fXh+WuPe6/MKk+tJuoBQTGVQ5ryFmomsNiwhwtLbQ44fLVHhyqEKSEdo/107xvbzhjmY/MAzn1Pmc9rd+OhFsjUCvgqI8cWNc/E694eJqg3J2S+I6YRzG3d2tR7laUivf+J38c2XmwSyXfdRoJpyZ9TixubpPk04WSchdFlEkxPBGEWLDkWOVL1PG5ztY48di7EzM1tvAwiT1BOxl4WRZ78Ewc+C5BVHwT658rIrcKJXXI/zBMsoReQT9xsRhpozbb576wNXggJdZsd2ysQY0O6Pihz54emwigm+zPbO5n8HvlrGKf6dSsrwusUJ1BIY4wI6qjz7gweRryReDEvEzMT8Ul4mIrigRy4yL2w+03qAclz8oGwxinMvcu8vJzXg+uRm/WbOgyco4gTPQiN4NcXbzwhVtJlNWZYXCiiMb/i6IXuOzZmSjI7LqxLubD9RgOy/2890RLvVJQBBVnOowW8q+iE93CoVBr1l5D54opLS9fHYcM7ezV0Ul34qMu6K0uoBG0+aLVlZHKEecN2/VE4fh0zYEDaeqRZfNH2gnAGmokdmPtEHlp33pvJ0IFDAbxKq2CVFFdB+lCGlaLQuZ5v6Mhq4b6H8DjaGZqo/vcB/MK4pr/F1SRjLzSHyh7Ey4ogBYSOXWfaeXQiZZFoEfxIUG9PzofIA1CCFk+eZSG7bGY4wXe2Whhh5bs+cJ3duYI9SL+49WBABgB',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the page I fetched, the main framework it mentions and compares itself to is **FastAPI**. The page states that "FastAPI revolutionized web development by offering an innovative and ergonomic design" and that Pydantic AI was built with the aim "to bring that FastAPI feeling to GenAI app and agent development."

The page also mentions several other frameworks and libraries including:
- LangChain
- LlamaIndex  \n\
- AutoGPT
- Transformers
- CrewAI
- Instructor

It notes that "virtually every Python agent framework and LLM library" uses Pydantic Validation, which is the foundation that Pydantic AI builds upon.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=6346,
                    output_tokens=354,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 6346,
                        'output_tokens': 354,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_web_fetch_tool_stream(
    allow_model_requests: None, anthropic_api_key: str
):  # pragma: lax no cover
    from pydantic_ai.messages import PartDeltaEvent, PartStartEvent

    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebFetchTool()], model_settings=settings)

    # Iterate through the stream to ensure streaming code paths are covered
    event_parts: list[Any] = []
    async with agent.iter(  # pragma: lax no cover
        user_prompt='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    ) as agent_run:
        async for node in agent_run:  # pragma: lax no cover
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):  # pragma: lax no cover
                async with node.stream(agent_run.ctx) as request_stream:  # pragma: lax no cover
                    async for event in request_stream:  # pragma: lax no cover
                        if (  # pragma: lax no cover
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or isinstance(event, PartDeltaEvent):
                            event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
    )

    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants me to fetch the content from the URL https://ai.pydantic.dev and provide only the first sentence from that page. I need to use the web_fetch tool to get the content from this URL.',
                        signature='EusCCkYICRgCKkAG/7zhRcmUoiMtml5iZUXVv3nqupp8kgk0nrq9zOoklaXzVCnrb9kwLNWGETIcCaAnLd0cd0ESwjslkVKdV9n8EgxKKdu8LlEvh9VGIWIaDAJ2Ja2NEacp1Am6jSIwyNO36tV+Sj+q6dWf79U+3KOIa1khXbIYarpkIViCuYQaZwpJ4Vtedrd7dLWTY2d5KtIB9Pug5UPuvepSOjyhxLaohtGxmdvZN8crGwBdTJYF9GHSli/rzvkR6CpH+ixd8iSopwFcsJgQ3j68fr/yD7cHmZ06jU3LaESVEBwTHnlK0ABiYnGvD3SvX6PgImMSQxQ1ThARFTA7DePoWw+z5DI0L2vgSun2qTYHkmGxzaEskhNIBlK9r7wS3tVcO0Di4lD/rhYV61tklL2NBWJqvm7ZCtJTN09CzPFJy7HDkg7bSINVL4kuu9gTWEtb/o40tw1b+sO62UcfxQTVFQ4Cj8D8XFZbGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args='{"url": "https://ai.pydantic.dev"}',
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7244,
                    output_tokens=153,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7244,
                        'output_tokens': 153,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='The user wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me to fetch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from the URL https')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='://ai.pydantic.dev')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and provide')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' only')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the first sentence from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that page.')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' I need to use the web_fetch'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tool to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' get the content from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this URL.')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='EusCCkYICRgCKkAG/7zhRcmUoiMtml5iZUXVv3nqupp8kgk0nrq9zOoklaXzVCnrb9kwLNWGETIcCaAnLd0cd0ESwjslkVKdV9n8EgxKKdu8LlEvh9VGIWIaDAJ2Ja2NEacp1Am6jSIwyNO36tV+Sj+q6dWf79U+3KOIa1khXbIYarpkIViCuYQaZwpJ4Vtedrd7dLWTY2d5KtIB9Pug5UPuvepSOjyhxLaohtGxmdvZN8crGwBdTJYF9GHSli/rzvkR6CpH+ixd8iSopwFcsJgQ3j68fr/yD7cHmZ06jU3LaESVEBwTHnlK0ABiYnGvD3SvX6PgImMSQxQ1ThARFTA7DePoWw+z5DI0L2vgSun2qTYHkmGxzaEskhNIBlK9r7wS3tVcO0Di4lD/rhYV61tklL2NBWJqvm7ZCtJTN09CzPFJy7HDkg7bSINVL4kuu9gTWEtb/o40tw1b+sO62UcfxQTVFQ4Cj8D8XFZbGAE='
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(tool_name='web_fetch', tool_call_id=IsStr(), provider_name='anthropic'),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"url": "', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='https://ai', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='.p', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='yd', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='antic.dev"}', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    content={
                        'content': {
                            'citations': None,
                            'source': {
                                'data': '''\
Pydantic AI
GenAI Agent Framework, the Pydantic way
Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.
FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.
Yet despite virtually every Python agent framework and LLM library using Pydantic Validation, when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.
We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development.
Why use Pydantic AI
-
Built by the Pydantic Team:
[Pydantic Validation](https://docs.pydantic.dev/latest/)is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more. Why use the derivative when you can go straight to the source? -
Model-agnostic: Supports virtually every
[model](models/overview/)and provider: OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, and Perplexity; Azure AI Foundry, Amazon Bedrock, Google Vertex AI, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, and Outlines. If your favorite model or provider is not listed, you can easily implement a[custom model](models/overview/#custom-models). -
Seamless Observability: Tightly
[integrates](logfire/)with[Pydantic Logfire](https://pydantic.dev/logfire), our general-purpose OpenTelemetry observability platform, for real-time debugging, evals-based performance monitoring, and behavior, tracing, and cost tracking. If you already have an observability platform that supports OTel, you can[use that too](logfire/#alternative-observability-backends). -
Fully Type-safe: Designed to give your IDE or AI coding agent as much context as possible for auto-completion and
[type checking](agents/#static-type-checking), moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel. -
Powerful Evals: Enables you to systematically test and
[evaluate](evals/)the performance and accuracy of the agentic systems you build, and monitor the performance over time in Pydantic Logfire. -
MCP, A2A, and UI: Integrates the
[Model Context Protocol](mcp/overview/),[Agent2Agent](a2a/), and various[UI event stream](ui/overview/)standards to give your agent access to external tools and data, let it interoperate with other agents, and build interactive applications with streaming event-based communication. -
Human-in-the-Loop Tool Approval: Easily lets you flag that certain tool calls
[require approval](deferred-tools/#human-in-the-loop-tool-approval)before they can proceed, possibly depending on tool call arguments, conversation history, or user preferences. -
Durable Execution: Enables you to build
[durable agents](durable_execution/overview/)that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. -
Streamed Outputs: Provides the ability to
[stream](output/#streamed-results)structured output continuously, with immediate validation, ensuring real time access to generated data. -
Graph Support: Provides a powerful way to define
[graphs](graph/)using type hints, for use in complex applications where standard control flow can degrade to spaghetti code.
Realistically though, no list is going to be as convincing as [giving it a try](#next-steps) and seeing how it makes you feel!
Sign up for our newsletter, The Pydantic Stack, with updates & tutorials on Pydantic AI, Logfire, and Pydantic:
Hello World Example
Here's a minimal example of Pydantic AI:
[Learn about Gateway](gateway)hello_world.py
from pydantic_ai import Agent
agent = Agent( # (1)!
'gateway/anthropic:claude-sonnet-4-0',
instructions='Be concise, reply with one sentence.', # (2)!
)
result = agent.run_sync('Where does "hello world" come from?') # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
- We configure the agent to use
[Anthropic's Claude Sonnet 4.0](api/models/anthropic/)model, but you can also set the model when running the agent. - Register static
[instructions](agents/#instructions)using a keyword argument to the agent. [Run the agent](agents/#running-agents)synchronously, starting a conversation with the LLM.
from pydantic_ai import Agent
agent = Agent( # (1)!
'anthropic:claude-sonnet-4-0',
instructions='Be concise, reply with one sentence.', # (2)!
)
result = agent.run_sync('Where does "hello world" come from?') # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
- We configure the agent to use
[Anthropic's Claude Sonnet 4.0](api/models/anthropic/)model, but you can also set the model when running the agent. - Register static
[instructions](agents/#instructions)using a keyword argument to the agent. [Run the agent](agents/#running-agents)synchronously, starting a conversation with the LLM.
(This example is complete, it can be run "as is", assuming you've [installed the pydantic_ai package](install/))
The exchange will be very short: Pydantic AI will send the instructions and the user prompt to the LLM, and the model will return a text response.
Not very interesting yet, but we can easily add [tools](tools/), [dynamic instructions](agents/#instructions), and [structured outputs](output/) to build more powerful agents.
Tools & Dependency Injection Example
Here is a concise example using Pydantic AI to build a support agent for a bank:
[Learn about Gateway](gateway)bank_support.py
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
@dataclass
class SupportDependencies: # (3)!
customer_id: int
db: DatabaseConn # (12)!
class SupportOutput(BaseModel): # (13)!
support_advice: str = Field(description='Advice returned to the customer')
block_card: bool = Field(description="Whether to block the customer's card")
risk: int = Field(description='Risk level of query', ge=0, le=10)
support_agent = Agent( # (1)!
'gateway/openai:gpt-5', # (2)!
deps_type=SupportDependencies,
output_type=SupportOutput, # (9)!
instructions=( # (4)!
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
@support_agent.instructions # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
return f"The customer's name is {customer_name!r}"
@support_agent.tool # (6)!
async def customer_balance(
ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
"""Returns the customer's current account balance.""" # (7)!
return await ctx.deps.db.customer_balance(
id=ctx.deps.customer_id,
include_pending=include_pending,
)
... # (11)!
async def main():
deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = await support_agent.run('What is my balance?', deps=deps) # (8)!
print(result.output) # (10)!
"""
support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
"""
result = await support_agent.run('I just lost my card!', deps=deps)
print(result.output)
"""
support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
"""
- This
[agent](agents/)will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has typeAgent[SupportDependencies, SupportOutput]
. - Here we configure the agent to use
[OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent. - The
SupportDependencies
dataclass is used to pass data, connections, and logic into the model that will be needed when running[instructions](agents/#instructions)and[tool](tools/)functions. Pydantic AI's system of dependency injection provides a[type-safe](agents/#static-type-checking)way to customise the behavior of your agents, and can be especially useful when running[unit tests](testing/)and evals. - Static
[instructions](agents/#instructions)can be registered with theto the agent.instructions
keyword argument - Dynamic
[instructions](agents/#instructions)can be registered with thedecorator, and can make use of dependency injection. Dependencies are carried via the@agent.instructions
argument, which is parameterized with theRunContext
deps_type
from above. If the type annotation here is wrong, static type checkers will catch it. - The
decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via@agent.tool
, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.RunContext
- The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are
[extracted](tools/#function-tools-and-schema)from the docstring and added to the parameter schema sent to the LLM. [Run the agent](agents/#running-agents)asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.- The response from the agent will be guaranteed to be a
SupportOutput
. If validation fails[reflection](agents/#reflection-and-self-correction), the agent is prompted to try again. - The output will be validated with Pydantic to guarantee it is a
SupportOutput
, since the agent is generic, it'll also be typed as aSupportOutput
to aid with static type checking. - In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
- This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
- This
[Pydantic](https://docs.pydantic.dev)model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
@dataclass
class SupportDependencies: # (3)!
customer_id: int
db: DatabaseConn # (12)!
class SupportOutput(BaseModel): # (13)!
support_advice: str = Field(description='Advice returned to the customer')
block_card: bool = Field(description="Whether to block the customer's card")
risk: int = Field(description='Risk level of query', ge=0, le=10)
support_agent = Agent( # (1)!
'openai:gpt-5', # (2)!
deps_type=SupportDependencies,
output_type=SupportOutput, # (9)!
instructions=( # (4)!
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
@support_agent.instructions # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
return f"The customer's name is {customer_name!r}"
@support_agent.tool # (6)!
async def customer_balance(
ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
"""Returns the customer's current account balance.""" # (7)!
return await ctx.deps.db.customer_balance(
id=ctx.deps.customer_id,
include_pending=include_pending,
)
... # (11)!
async def main():
deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = await support_agent.run('What is my balance?', deps=deps) # (8)!
print(result.output) # (10)!
"""
support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
"""
result = await support_agent.run('I just lost my card!', deps=deps)
print(result.output)
"""
support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
"""
- This
[agent](agents/)will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has typeAgent[SupportDependencies, SupportOutput]
. - Here we configure the agent to use
[OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent. - The
SupportDependencies
dataclass is used to pass data, connections, and logic into the model that will be needed when running[instructions](agents/#instructions)and[tool](tools/)functions. Pydantic AI's system of dependency injection provides a[type-safe](agents/#static-type-checking)way to customise the behavior of your agents, and can be especially useful when running[unit tests](testing/)and evals. - Static
[instructions](agents/#instructions)can be registered with theto the agent.instructions
keyword argument - Dynamic
[instructions](agents/#instructions)can be registered with thedecorator, and can make use of dependency injection. Dependencies are carried via the@agent.instructions
argument, which is parameterized with theRunContext
deps_type
from above. If the type annotation here is wrong, static type checkers will catch it. - The
decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via@agent.tool
, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.RunContext
- The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are
[extracted](tools/#function-tools-and-schema)from the docstring and added to the parameter schema sent to the LLM. [Run the agent](agents/#running-agents)asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.- The response from the agent will be guaranteed to be a
SupportOutput
. If validation fails[reflection](agents/#reflection-and-self-correction), the agent is prompted to try again. - The output will be validated with Pydantic to guarantee it is a
SupportOutput
, since the agent is generic, it'll also be typed as aSupportOutput
to aid with static type checking. - In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
- This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
- This
[Pydantic](https://docs.pydantic.dev)model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.
Complete bank_support.py
example
The code included here is incomplete for the sake of brevity (the definition of DatabaseConn
is missing); you can find the complete bank_support.py
example [here](examples/bank-support/).
Instrumentation with Pydantic Logfire
Even a simple agent with just a handful of tools can result in a lot of back-and-forth with the LLM, making it nearly impossible to be confident of what's going on just from reading the code. To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.
To do this, we need to [set up Logfire](logfire/#using-logfire), and add the following to our code:
[Learn about Gateway](gateway)bank_support_with_logfire.py
...
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
import logfire
logfire.configure() # (1)!
logfire.instrument_pydantic_ai() # (2)!
logfire.instrument_asyncpg() # (3)!
...
support_agent = Agent(
'gateway/openai:gpt-5',
deps_type=SupportDependencies,
output_type=SupportOutput,
system_prompt=(
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
- Configure the Logfire SDK, this will fail if project is not set up.
- This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the
to the agent.instrument=True
keyword argument - In our demo,
DatabaseConn
usesto connect to a PostgreSQL database, soasyncpg
is used to log the database queries.logfire.instrument_asyncpg()
...
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
import logfire
logfire.configure() # (1)!
logfire.instrument_pydantic_ai() # (2)!
logfire.instrument_asyncpg() # (3)!
...
support_agent = Agent(
'openai:gpt-5',
deps_type=SupportDependencies,
output_type=SupportOutput,
system_prompt=(
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
- Configure the Logfire SDK, this will fail if project is not set up.
- This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the
to the agent.instrument=True
keyword argument - In our demo,
DatabaseConn
usesto connect to a PostgreSQL database, soasyncpg
is used to log the database queries.logfire.instrument_asyncpg()
That's enough to get the following view of your agent in action:
See [Monitoring and Performance](logfire/) to learn more.
llms.txt
The Pydantic AI documentation is available in the [llms.txt](https://llmstxt.org/) format.
This format is defined in Markdown and suited for LLMs and AI coding assistants and agents.
Two formats are available:
: a file containing a brief description of the project, along with links to the different sections of the documentation. The structure of this file is described in detailsllms.txt
[here](https://llmstxt.org/#format).: Similar to thellms-full.txt
llms.txt
file, but every link content is included. Note that this file may be too large for some LLMs.
As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.
Next Steps
To try Pydantic AI for yourself, [install it](install/) and follow the instructions [in the examples](examples/setup/).
Read the [docs](agents/) to learn more about building applications with Pydantic AI.
Read the [API Reference](api/agent/) to understand Pydantic AI's interface.
Join [ Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [ GitHub](https://github.com/pydantic/pydantic-ai/issues) if you have any questions.\
''',
                                'media_type': 'text/plain',
                                'type': 'text',
                            },
                            'title': 'Pydantic AI',
                            'type': 'document',
                        },
                        'retrieved_at': IsStr(),
                        'type': 'web_fetch_result',
                        'url': 'https://ai.pydantic.dev',
                    },
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ydantic AI is a')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' agent')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' framework')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' designe')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d to help')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' you quickly')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' confi')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='dently,')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and pain')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='lessly build production')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' grade')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' applications')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d workflows')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Gener')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ative AI.')),
        ]
    )


async def test_anthropic_web_fetch_tool_message_replay():
    """Test that BuiltinToolCallPart and BuiltinToolReturnPart for WebFetchTool are correctly serialized."""
    from typing import cast

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Create message history with BuiltinToolCallPart and BuiltinToolReturnPart
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Test')], timestamp=IsDatetime()),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    provider_name=m.system,
                    tool_name=WebFetchTool.kind,
                    args={'url': 'https://example.com'},
                    tool_call_id='test_id_1',
                ),
                BuiltinToolReturnPart(
                    provider_name=m.system,
                    tool_name=WebFetchTool.kind,
                    content={
                        'content': {'type': 'document'},
                        'type': 'web_fetch_result',
                        'url': 'https://example.com',
                        'retrieved_at': '2025-01-01T00:00:00Z',
                    },
                    tool_call_id='test_id_1',
                ),
            ],
            model_name='claude-sonnet-4-0',
        ),
    ]

    # Call _map_message to trigger serialization
    model_settings = {}
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[WebFetchTool()],
        output_tools=[],
    )

    system_prompt, anthropic_messages = await m._map_message(messages, model_request_parameters, model_settings)  # pyright: ignore[reportPrivateUsage,reportArgumentType]

    # Verify the messages were serialized correctly
    assert system_prompt is None or isinstance(system_prompt, (list | str))
    assert len(anthropic_messages) == 2
    assert anthropic_messages[1]['role'] == 'assistant'

    # Check that server_tool_use block is present
    content = anthropic_messages[1]['content']
    assert any(
        isinstance(item, dict) and item.get('type') == 'server_tool_use' and item.get('name') == 'web_fetch'
        for item in content
    )

    # Check that web_fetch_tool_result block is present and contains URL and retrieved_at
    web_fetch_result = next(
        item for item in content if isinstance(item, dict) and item.get('type') == 'web_fetch_tool_result'
    )
    assert 'content' in web_fetch_result
    result_content = web_fetch_result['content']
    assert isinstance(result_content, dict)  # Type narrowing for mypy
    assert result_content['type'] == 'web_fetch_result'  # type: ignore[typeddict-item]
    assert result_content['url'] == 'https://example.com'  # type: ignore[typeddict-item]
    # retrieved_at is optional - cast to avoid complex union type issues
    assert cast(dict, result_content).get('retrieved_at') == '2025-01-01T00:00:00Z'  # pyright: ignore[reportUnknownMemberType,reportMissingTypeArgument]
    assert 'content' in result_content  # The actual document content


async def test_anthropic_web_fetch_tool_with_parameters():
    """Test that WebFetchTool parameters are correctly passed to Anthropic API."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Create WebFetchTool with all parameters
    web_fetch_tool = WebFetchTool(
        max_uses=5,
        allowed_domains=['example.com', 'ai.pydantic.dev'],
        enable_citations=True,
        max_content_tokens=50000,
    )

    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[web_fetch_tool],
        output_tools=[],
    )

    # Get tools from model
    tools, _, _ = m._add_builtin_tools([], model_request_parameters)  # pyright: ignore[reportPrivateUsage]
    # Find the web_fetch tool
    web_fetch_tool_param = next((t for t in tools if t.get('name') == 'web_fetch'), None)
    assert web_fetch_tool_param is not None

    # Verify all parameters are passed correctly
    assert web_fetch_tool_param.get('type') == 'web_fetch_20250910'
    assert web_fetch_tool_param.get('max_uses') == 5
    assert web_fetch_tool_param.get('allowed_domains') == ['example.com', 'ai.pydantic.dev']
    assert web_fetch_tool_param.get('blocked_domains') is None
    assert web_fetch_tool_param.get('citations') == {'enabled': True}
    assert web_fetch_tool_param.get('max_content_tokens') == 50000


async def test_anthropic_web_fetch_tool_domain_filtering():
    """Test that blocked_domains work and are mutually exclusive with allowed_domains."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Test with blocked_domains
    web_fetch_tool = WebFetchTool(blocked_domains=['private.example.com', 'internal.example.com'])

    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[web_fetch_tool],
        output_tools=[],
    )

    # Get tools from model
    tools, _, _ = m._add_builtin_tools([], model_request_parameters)  # pyright: ignore[reportPrivateUsage]
    # Find the web_fetch tool
    web_fetch_tool_param = next((t for t in tools if t.get('name') == 'web_fetch'), None)
    assert web_fetch_tool_param is not None

    # Verify blocked_domains is passed correctly
    assert web_fetch_tool_param.get('blocked_domains') == ['private.example.com', 'internal.example.com']
    assert web_fetch_tool_param.get('allowed_domains') is None


@pytest.mark.vcr()
async def test_anthropic_mcp_servers(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
            )
        ],
        model_settings=settings,
    )

    result = await agent.run('Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic/pydantic-ai repository and wants me to keep the answer short. I should use the deepwiki tools to get information about this repository. Let me start by asking a general question about what this repository is about.',
                        signature='EqUDCkYICBgCKkCTiLjx5Rzw9zXo4pFDhFAc9Ci1R+d2fpkiqw7IPt1PgxBankr7bhRfh2iQOFEUy7sYVtsBxvnHW8zfBRxH1j6lEgySvdOyObrcFdJX3qkaDMAMCdLHIevZ/mSx/SIwi917U34N5jLQH1yMoCx/k72klLG5v42vcwUTG4ngKDI69Ddaf0eeDpgg3tL5FHfvKowCnslWg3Pd3ITe+TLlzu+OVZhRKU9SEwDJbjV7ZF954Ls6XExAfjdXhrhvXDB+hz6fZFPGFEfXV7jwElFT5HcGPWy84xvlwzbklZ2zH3XViik0B5dMErMAKs6IVwqXo3s+0p9xtX5gCBuvLkalET2upNsmdKGJv7WQWoaLch5N07uvSgWkO8AkGuVtBgqZH+uRGlPfYlnAgifNHu00GSAVK3beeyZfpnSQ6LQKcH+wVmrOi/3UvzA5f1LvsXG32gQKUCxztATnlBaI+7GMs1IAloaRHBndyRoe8Lwv79zZe9u9gnF9WCgK3yQsAR5hGZXlBKiIWfnRrXQ7QmA2hVO+mhEOCnz7OQkMIEUlfxgB',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'What is pydantic-ai and what does this repository do?',
                            },
                        },
                        tool_call_id='mcptoolu_01SAss3KEwASziHZoMR6HcZU',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': IsStr(),
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01SAss3KEwASziHZoMR6HcZU',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic AI** is a Python agent framework for building production-grade applications with Generative AI. It provides:

- **Type-safe agents** with compile-time validation using `Agent[Deps, Output]`
- **Model-agnostic design** supporting 15+ LLM providers (OpenAI, Anthropic, Google, etc.)
- **Structured outputs** with automatic Pydantic validation and self-correction
- **Built-in observability** via OpenTelemetry and Logfire integration
- **Production tooling** including evaluation framework, durable execution, and tool system

The repo is organized as a monorepo with core packages like `pydantic-ai-slim` (core framework), `pydantic-graph` (execution engine), and `pydantic-evals` (evaluation tools). It emphasizes developer ergonomics and type safety, similar to Pydantic and FastAPI.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2674,
                    output_tokens=373,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2674,
                        'output_tokens': 373,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01MYDjkvBDRaKsY6PDwQz3n6',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'How about the pydantic repo in the same org?', message_history=messages
    )  # pragma: lax no cover
    messages = result.new_messages()  # pragma: lax no cover
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How about the pydantic repo in the same org?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic repo in the same org, so that would be pydantic/pydantic. I should ask about what this repository does and provide a short answer.',
                        signature='EtECCkYICBgCKkAkKy+K3Z/q4dGwZGr1MdsH8HLaULElUSaa/Y8A1L/Jp7y1AfJd1zrTL7Zfa2KoPr0HqO/AI/cJJreheuwcn/dWEgw0bPLie900a4h9wS0aDACnsdbr+adzpUyExiIwyuNjV82BVkK/kU+sMyrfbhgb6ob/DUgudJPaK5zR6cINAAGQnIy3iOXTwu3OUfPAKrgBzF9HD5HjiPSJdsxlkI0RA5Yjiol05/hR3fUB6WWrs0aouxIzlriJ6NzmzvqctkFJdRgAL9Mh06iK1A61PLyBWRdo1f5TBziFP1c6z7iQQzH9DdcaHvG8yLoaadbyTxMvTn2PtfEcSPjuZcLgv7QcF+HZXbDVjsHJW78OK2ta0M6/xuU1p4yG3qgoss3b0G6fAyvUVgVbb1wknkE/9W9gd2k/ZSh4P7F6AcvLTXQScTyMfWRtAWQqABgB',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic',
                                'question': 'What is Pydantic and what does this repository do?',
                            },
                        },
                        tool_call_id='mcptoolu_01A9RvAqDeoUnaMgQc6Nn75y',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': """\
Pydantic is a Python library for data validation, parsing, and serialization using type hints  . This repository, `pydantic/pydantic`, contains the source code for the Pydantic library itself, including its core validation logic, documentation, and continuous integration/continuous deployment (CI/CD) pipelines  .

## What is Pydantic

Pydantic is designed to ensure that data conforms to specified types and constraints at runtime . It leverages Python type hints to define data schemas and provides mechanisms for data conversion and validation . The library's core validation logic is implemented in Rust within a separate package called `pydantic-core`, which contributes to its performance .

Pydantic offers several user-facing APIs for validation:
*   `BaseModel`: Used for defining class-based models with fields, suitable for domain models, API schemas, and configuration .
*   `TypeAdapter`: Provides a flexible way to validate and serialize arbitrary Python types, including primitive types and dataclasses .
*   `@dataclass`: Enhances Python's built-in dataclasses with Pydantic's validation capabilities .
*   `@validate_call`: Used for validating function arguments and return values .

## What this Repository Does

The `pydantic/pydantic` repository serves as the development hub for the Pydantic library. Its primary functions include:

### Core Library Development
The repository contains the Python source code for the Pydantic library, including modules for `BaseModel` , `Field` definitions , configuration management , and type adapters . It also includes internal modules responsible for model construction and schema generation .

### Documentation
The repository hosts the documentation for Pydantic, which is built using MkDocs . The documentation covers installation instructions , core concepts like models , fields, and JSON Schema generation . It also includes information on contributing to the project .

### Continuous Integration and Deployment (CI/CD)
The repository utilizes GitHub Actions for its CI/CD pipeline . This pipeline includes:
*   **Linting**: Checks code quality and style .
*   **Testing**: Runs a comprehensive test suite across multiple operating systems and Python versions . This includes memory profiling tests, Mypy plugin tests, and type-checking integration tests   .
*   **Coverage**: Aggregates test coverage data and posts comments to pull requests .
*   **Release Process**: Automates publishing new versions to PyPI and sending release announcements .
*   **Third-Party Integration Testing**: Tests Pydantic's compatibility with other popular libraries like FastAPI, SQLModel, and Beanie .
*   **Dependency Management**: Uses `uv` for managing dependencies and includes workflows to check compatibility with various dependency versions  .
*   **Performance Benchmarking**: Utilizes CodSpeed to track and analyze performance .

## Versioning and Compatibility
Pydantic maintains strict version compatibility between the pure Python package (`pydantic`) and its Rust-based validation core (`pydantic-core`)  . A `SystemError` is raised if there's a mismatch in `pydantic-core` versions, ensuring a stable environment . The `version_info()` function provides detailed version information for Pydantic and its dependencies .

Notes:
The `CITATION.cff` file also provides a concise description of Pydantic as "the most widely used data validation library for Python" . The `README.md` and `docs/index.md` files reiterate this, emphasizing its speed and extensibility  .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic)](/wiki/pydantic/pydantic#1)
- [Development and Deployment (pydantic/pydantic)](/wiki/pydantic/pydantic#7)

View this search on DeepWiki: https://deepwiki.com/search/what-is-pydantic-and-what-does_dab96efa-752a-4688-a630-3f4658084a88
""",
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01A9RvAqDeoUnaMgQc6Nn75y',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic** is Python's most widely used data validation library for parsing, validation, and serialization using type hints. The repository contains:

**Core Features:**
- **Data validation** with automatic type conversion and constraint checking
- **Multiple APIs**: `BaseModel` for class-based models, `TypeAdapter` for arbitrary types, `@dataclass` decorator, and `@validate_call` for functions
- **High performance** via Rust-based validation core (`pydantic-core`)
- **JSON Schema generation** and comprehensive serialization support

**Repository Contents:**
- Python source code for the main Pydantic library
- Comprehensive documentation built with MkDocs
- Extensive CI/CD pipeline with testing across multiple Python versions and OS
- Integration testing with popular libraries (FastAPI, SQLModel, etc.)
- Performance benchmarking and dependency compatibility checks

Pydantic ensures runtime data integrity through type hints and is foundational to many Python frameworks, especially in web APIs and data processing applications.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=5262,
                    output_tokens=369,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 5262,
                        'output_tokens': 369,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01DSGib8F7nNoYprfYSGp1sd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_mcp_servers_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                allowed_tools=['ask_question'],
            )
        ],
        model_settings=settings,
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or (isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)):
                            event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic/pydantic-ai repository. They want a short answer about the repo. I should use the deepwiki_ask_question function to get information about this repository.',
                        signature='EuoCCkYICBgCKkDPqznnPHupi9rVXvaQQqrMprXof9wtQsCqw7Yw687UIk/FvF65omU22QO+CmIcYqTwhBfifPEp9A3/lM9C8cIcEgzGsjorcyNe2H0ZFf8aDCA4iLG6qgUL6fLhzCIwVWcg65CrvSFusXtMH18p+XiF+BUxT+rvnCFsnLbFsxtjGyKh1j4UW6V0Tk0O7+3sKtEBEzvxztXkMkeXkXRsQFJ00jTNhkUHu74sqnh6QxgV8wK2vlJRnBnes/oh7QdED0h/pZaUbxplYJiPFisWx/zTJQvOv29I46sM2CdY5ggGO1KWrEF/pognyod+jdCdb481XUET9T7nl/VMz/Og2QkyGf+5MvSecKQhujlS0VFhCgaYv68sl0Fv3hj2AkeE4vcYu3YdDaNDLXerbIaLCMkkn08NID/wKZTwtLSL+N6+kOi+4peGqXDNps8oa3mqIn7NAWFlwEUrFZd5kjtDkQ5dw/IYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args='{"action":"call_tool","tool_name":"ask_question","tool_args":{"repoName": "pydantic/pydantic-ai", "question": "What is this repository about? What are its main features and purpose?"}}',
                        tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': IsStr(),
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic-AI** is a framework for building Generative AI applications with type safety. It provides:

- **Unified LLM interface** - Works with OpenAI, Anthropic, Google, Groq, Cohere, Mistral, AWS Bedrock, and more
- **Type-safe agents** - Uses Pydantic for validation and type checking throughout
- **Tool integration** - Easily add custom functions/tools agents can call
- **Graph-based execution** - Manages agent workflows as finite state machines
- **Multiple output formats** - Text, structured data, and multimodal content
- **Durable execution** - Integration with systems like DBOS and Temporal for fault tolerance
- **Streaming support** - Stream responses in real-time

It's designed to simplify building robust, production-ready AI agents while abstracting away provider-specific complexities.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3042,
                    output_tokens=354,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 3042,
                        'output_tokens': 354,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Xf6SmUVY1mDrSwFc5RsY3n',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                    provider_name='anthropic',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"action":"call_tool","tool_name":"ask_question","tool_args":',
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                ),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"repoName"', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=': "', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='pydantic', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='/pydantic-ai', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='"', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta=', "question', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='": "What', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=' is ', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='this repo', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='sitory about', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='? Wha', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='t are i', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='ts main feat', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='ure', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='s and purpo', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='se?"}', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='}', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'content': [
                            {
                                'citations': None,
                                'text': """\
This repository, `pydantic/pydantic-ai`, is a GenAI Agent Framework that leverages Pydantic for building Generative AI applications. Its main purpose is to provide a unified and type-safe way to interact with various large language models (LLMs) from different providers, manage agent execution flows, and integrate with external tools and services. \n\

## Main Features and Purpose

The `pydantic-ai` repository offers several core features:

### 1. Agent System
The `Agent` class serves as the main orchestrator for managing interactions with LLMs and executing tasks.  Agents can be configured with generic types for dependency injection (`Agent[AgentDepsT, OutputDataT]`) and output validation, ensuring type safety throughout the application. \n\

Agents support various execution methods:
*   `agent.run()`: An asynchronous function that returns a completed `RunResult`. \n\
*   `agent.run_sync()`: A synchronous function that internally calls `run()` to return a completed `RunResult`. \n\
*   `agent.run_stream()`: An asynchronous context manager for streaming text and structured output. \n\
*   `agent.run_stream_events()`: Returns an asynchronous iterable of `AgentStreamEvent`s and a final `AgentRunResultEvent`. \n\
*   `agent.iter()`: A context manager that provides an asynchronous iterable over the nodes of the agent's underlying `Graph`, allowing for deeper control and insight into the execution flow. \n\

### 2. Model Integration
The framework provides a unified interface for integrating with various LLM providers, including OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, and HuggingFace.  Each model integration follows a consistent settings pattern with provider-specific prefixes (e.g., `google_*`, `anthropic_*`). \n\

Examples of supported models and their capabilities include:
*   `GoogleModel`: Integrates with Google's Gemini API, supporting both Gemini API (`google-gla`) and Vertex AI (`google-vertex`) providers.  It supports token counting, streaming, built-in tools like `WebSearchTool`, `WebFetchTool`, `CodeExecutionTool`, and native JSON schema output. \n\
*   `AnthropicModel`: Uses Anthropic's beta API for advanced features like "Thinking Blocks" and built-in tools. \n\
*   `GroqModel`: Offers high-speed inference and specialized reasoning support with configurable reasoning formats. \n\
*   `MistralModel`: Supports customizable JSON schema prompting and thinking support. \n\
*   `BedrockConverseModel`: Utilizes AWS Bedrock's Converse API for unified access to various foundation models like Claude, Titan, Llama, and Mistral. \n\
*   `CohereModel`: Integrates with Cohere's v2 API for chat completions, including thinking support and tool calling. \n\

The framework also supports multimodal inputs such as `AudioUrl`, `DocumentUrl`, `ImageUrl`, and `VideoUrl`, allowing agents to process and respond to diverse content types. \n\

### 3. Graph-based Execution
Pydantic AI uses `pydantic-graph` to manage the execution flow of agents, representing it as a finite state machine.  The execution typically flows through `UserPromptNode` → `ModelRequestNode` → `CallToolsNode`.  This allows for detailed tracking of message history and usage. \n\

### 4. Tool System
Function tools enable models to perform actions and retrieve additional information.  Tools can be registered using decorators like `@agent.tool` (for tools needing `RunContext` access) or `@agent.tool_plain` (for tools without `RunContext` access).  The framework also supports toolsets for managing collections of tools. \n\

Tools can return various types of output, including anything Pydantic can serialize to JSON, as well as multimodal content like `AudioUrl`, `VideoUrl`, `ImageUrl`, or `DocumentUrl`.  The `ToolReturn` object allows for separating the `return_value` (for the model), `content` (for additional context), and `metadata` (for application-specific use). \n\

Built-in tools like `WebFetchTool` allow agents to pull web content into their context. \n\

### 5. Output Handling
The framework supports various output types:
*   `TextOutput`: Plain text responses. \n\
*   `ToolOutput`: Structured data via tool calls. \n\
*   `NativeOutput`: Provider-specific structured output. \n\
*   `PromptedOutput`: Prompt-based structured extraction. \n\

### 6. Durable Execution
Pydantic AI integrates with durable execution systems like DBOS and Temporal.  This allows agents to maintain state and resume execution after failures or restarts, making them suitable for long-running or fault-tolerant applications. \n\

### 7. Multi-Agent Patterns and Integrations
The repository supports multi-agent applications and various integrations, including:
*   Pydantic Evals: For evaluating agent performance. \n\
*   Pydantic Graph: The underlying graph execution engine. \n\
*   Logfire: For debugging and monitoring. \n\
*   Agent-User Interaction (AG-UI) and Agent2Agent (A2A): For facilitating interactions between agents and users, and between agents themselves. \n\
*   Clai: A CLI tool. \n\

## Purpose

The overarching purpose of `pydantic-ai` is to simplify the development of robust and reliable Generative AI applications by providing a structured, type-safe, and extensible framework. It aims to abstract away the complexities of interacting with different LLM providers and managing agent workflows, allowing developers to focus on application logic. \n\

Notes:
The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, outlining development commands and project architecture.  The `mkdocs.yml` file defines the structure and content of the project's documentation, further detailing the features and organization of the repository. \n\

Wiki pages you might want to explore:
- [Google, Anthropic and Other Providers (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.3)

View this search on DeepWiki: https://deepwiki.com/search/what-is-this-repository-about_5104a64d-2f5e-4461-80d8-eb0892242441
""",
                                'type': 'text',
                            }
                        ],
                        'is_error': False,
                    },
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
        ]
    )


async def test_anthropic_code_execution_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        model_settings=settings,
        instructions='Always use the code execution tool for math.',
    )

    result = await agent.run('How much is 3 * 12390?')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How much is 3 * 12390?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the code execution tool for math.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking for a simple multiplication: 3 * 12390. This is a mathematical calculation, and according to my guidelines, I should always use the code execution tool for math. Even though this is a relatively simple calculation that could be done mentally, the instruction is clear that I should use the code execution tool for math.',
                        signature='EvsDCkYIBxgCKkCSFDXODoOrOHU14Yv7+TNxuR4sDsJKw9y9C1gGPIWqslF6apNZ1xwJ94E9KsQBfXlZ/ELoBSTj3YT0liwueN6kEgxrakXTN1a+YafcnckaDC2EYhQsezxdE/P7XSIwczAl/PquNGpiOLqC5DnYKvD2+F0JhBQsbLe1bQi/VR0XCQdd+4DZ5dBU5AmuDcntKuICIMg145F3vP8bFnTdUMOIQY0NASypKRnHj6owIkuqWJ+pwu6OdpDt2a+Lr7R1dw860hcPjEp65eg5nwtyi8bw1pzfQJmC48DoiQn/OYeiXMWeNv5HoKEK/lkikqVPcTnD03MytUsNGRqUBfDvr4bxNgxqeAENi5pZ21ySnjxhC879gN0G3uriEM8o4LXj/X2DotKO1lvIEL/2RQZGrFulDLq5I2FW51YBY3kzHerK7zwFgs3t39VLsy7Q3T6sLi4yh4BbFxF4RaSOCicTRbMYC8UO85uhArSSm/0EDDhX+kxIGJZ91F6Vv0vSS4qLy+55buZ8Jj4/P86t9YMxBeylQ/tUNGzhISqc1+CZeQ4aZKiRyQmlfkA6bcM42JAFQT/c0EbM2JmDsiSpkM8d021E9hqrr2eIhasaOo4vG5yUz7f9aSaRc/Muy02mckNxxxS7UshBCxr8veoMa0HYnB/rBNFeGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 3 * 12390
print(f"3 * 12390 = {result}")\
"""
                        },
                        tool_call_id='srvtoolu_01Pc4vcD1JPUDcVhHaskFUfn',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '3 * 12390 = 37170\n',
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_01Pc4vcD1JPUDcVhHaskFUfn',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='3 * 12390 = 37170'),
                ],
                usage=RequestUsage(
                    input_tokens=1771,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1771,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CTCwceSoRxi8Pf16Fb7Tn'},
                provider_response_id='msg_018bVTPr9khzuds31rFDuqW4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('How about 4 * 12390?')
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How about 4 * 12390?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the code execution tool for math.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking for a simple multiplication: 4 * 12390. This is a computational task that requires precise calculation, so I should use the code execution tool to get the accurate result.',
                        signature='EucCCkYIBxgCKkDrAwZF3dM/a2UiJFMD/+Z5mdZOkFXxJ1vmAg7GWzC2YUTBKtKvys1yFaWmkUuBSYBC/kaTPYVj28qa94V0Q/ngEgw+4333itH5QH/0B6gaDHxUZy/HGNpU04RbZiIwmQeS7P+gLHlV9b0tRYciwVbpjZl8WkrunyWyD5xXTC7bzv/tQKv8kMjxRsRGZZH1Ks4BDiNK1tuAlz4x5LDAsui8/8vBDY1c+NRtc6y0bOgxSXFXSemv2BHm7VokC7JG8+iCQEY9HIyFtyjLeJ93niDCszU8YHPtAa4o2Orw8K4Tc4Y18U/TqfgnZulkjkeONhDJP9uUk4Db4woJiLpAx13X8W5TriwqHWMRM2+D0coqTTWTovC/xbVFFZZmwyqaz/h6V6qqokyLpbqb+5B5kw/uQfybUv28h3GqxFyuD62zM9OPyMqbd2GrAPbSLE2JETkJsp6GzxVEh1vNI3DMgdQYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 4 * 12390
print(f"4 * 12390 = {result}")\
"""
                        },
                        tool_call_id='srvtoolu_017iCje5DPMZEdgBkxj1osgt',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '4 * 12390 = 49560\n',
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_017iCje5DPMZEdgBkxj1osgt',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='4 * 12390 = 49560'),
                ],
                usage=RequestUsage(
                    input_tokens=1741,
                    output_tokens=143,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1741,
                        'output_tokens': 143,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CTCwdXe48NC7LaX3rxQ4d'},
                provider_response_id='msg_01VngRFBcNddwrYQoKUmdePY',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_code_execution_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[CodeExecutionTool()], model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='what is 65465-6544 * 65464-6+1.02255') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to calculate a mathematical expression: 65465-6544 * 65464-6+1.02255

This involves multiplication and subtraction operations, and I need to be careful about the order of operations (PEMDAS/BODMAS). Let me break this down:

65465-6544 * 65464-6+1.02255

Following order of operations:
1. First, multiplication: 6544 * 65464
2. Then left to right for addition and subtraction: 65465 - (result from step 1) - 6 + 1.02255

This is a computational task that requires precise calculations, so I should use the code_execution tool to get an accurate result.\
""",
                        signature='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="I'll calculate this mathematical expression for you. Let me break it down step by step following the order of operations."
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                        tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
The answer to **65465-6544 * 65464-6+1.02255** is **-428,330,955.97745**.

Here's how it breaks down following the order of operations:
1. First, multiplication: 6,544 × 65,464 = 428,396,416
2. Then left to right: 65,465 - 428,396,416 = -428,330,951
3. Continue: -428,330,951 - 6 = -428,330,957
4. Finally: -428,330,957 + 1.02255 = -428,330,955.97745\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2316,
                    output_tokens=733,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2316,
                        'output_tokens': 733,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CTCwoJjSVr9b94gf8DPEi'},
                provider_response_id='msg_01TaPV5KLA8MsCPDuJNKPLF4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='The user is asking me to calculate'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' a mathematical expression: 65465-6544 *'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 65464-6+1.02255

This\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' involves multiplication and subtraction operations, and I need to be careful about the order of'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' operations (PEMDAS/BODMAS).'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 Let me break this down:

65\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='465-6544 * 65464-6+1.02255')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\


Following order of operations:
1. First, multiplication:\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 6544 * 65464')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\

2. Then left to right for\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' addition and subtraction: 65465'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' - (result from step 1)')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 - 6 + 1.02255

This\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' is a computational task that requires precise'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' calculations, so I should use the code_execution'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' tool to get an accurate result.'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ=='
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
The user is asking me to calculate a mathematical expression: 65465-6544 * 65464-6+1.02255

This involves multiplication and subtraction operations, and I need to be careful about the order of operations (PEMDAS/BODMAS). Let me break this down:

65465-6544 * 65464-6+1.02255

Following order of operations:
1. First, multiplication: 6544 * 65464
2. Then left to right for addition and subtraction: 65465 - (result from step 1) - 6 + 1.02255

This is a computational task that requires precise calculations, so I should use the code_execution tool to get an accurate result.\
""",
                    signature='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ==',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(content="I'll calculate this mathematical expression for you. Let me break"),
                previous_part_kind='thinking',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=1, delta=TextPartDelta(content_delta=' it down step by step following the order of operations.')
            ),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="I'll calculate this mathematical expression for you. Let me break it down step by step following the order of operations."
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=2, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG')
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=')\\n\\nexpression = \\"65465-6544 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='* 65464-6+1', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='.02255\\"\\nprint(f\\"Expression: {expression',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='}\\")\\n\\n# Let\'s break it down', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' step by step\\nstep1 = ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='6544 * 65464  ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='# Multiplication first\\nprint', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='(f\\"Step 1 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='- Multiplication: ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='6544 * 65464 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='= {step1}\\")\\n\\nstep2', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' = 65465 - step1  ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='# First subtraction\\nprint(f\\"Step', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' 2 - First subtraction:', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta=' 65465 - {step1', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='} = {step2}\\")\\n\\nstep', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='3 = step2 - 6  # Second subtraction\\nprint',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='(f\\"Step 3 - Second subtraction: {step2}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta=' - 6 = {step3', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='}\\")\\n\\nfinal_result = step3 + ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='1.02255  # Final addition\\nprint(f\\"Step ',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='4 - Final addition: {step3', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='} + 1.02255 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='= {final_result}\\")\\n\\n#', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=" Let's also verify with", tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' direct calculation\\ndirect_result = 65',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='465-6544 * 65464-', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='6+1.02255\\nprint(f\\"\\\\nDirect calculation:',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' {direct_result}\\")\\nprint', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='(f\\"Results match: {final_result == direct',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='_result}\\")', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG')
            ),
            PartEndEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={
                        'content': [],
                        'return_code': 0,
                        'stderr': '',
                        'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                        'type': 'code_execution_result',
                    },
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=4, part=TextPart(content='The answer to'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' **65465-6544 * ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='65464-6+1.02255** is **')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='-428,330,955.97745**.')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\


Here's how it breaks down following the order of operations:
1. First\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=', multiplication: 6,544 × 65,464 ')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
= 428,396,416
2. Then left\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' to right: 65,465 - 428')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
,396,416 = -428,330,951
3\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='. Continue: -428,330,951 -')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' 6 = -428,330')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
,957
4. Finally: -428,330,957 + \
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='1.02255 = -428,330,955.97745')),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content="""\
The answer to **65465-6544 * 65464-6+1.02255** is **-428,330,955.97745**.

Here's how it breaks down following the order of operations:
1. First, multiplication: 6,544 × 65,464 = 428,396,416
2. Then left to right: 65,465 - 428,396,416 = -428,330,951
3. Continue: -428,330,951 - 6 = -428,330,957
4. Finally: -428,330,957 + 1.02255 = -428,330,955.97745\
"""
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={
                        'content': [],
                        'return_code': 0,
                        'stderr': '',
                        'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                        'type': 'code_execution_result',
                    },
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


async def test_anthropic_shell_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        builtin_tools=[ShellTool()],
        instructions='Always use the shell tool. Keep responses brief.',
    )

    result = await agent.run('What is 2 + 2? Use python to calculate it.')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2? Use python to calculate it.', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='shell',
                        args={'command': 'python3 -c "print(2 + 2)"'},
                        tool_call_id='srvtoolu_018J4uFgCiKo2YeDS8DmVJtE',
                        provider_name='anthropic',
                        provider_details={'server_tool_name': 'bash_code_execution'},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='shell',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '4\n',
                            'type': 'bash_code_execution_result',
                        },
                        tool_call_id='srvtoolu_018J4uFgCiKo2YeDS8DmVJtE',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='2 + 2 = **4**'),
                ],
                usage=RequestUsage(
                    input_tokens=4627,
                    output_tokens=82,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 4627,
                        'output_tokens': 82,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CYvsF3eEuthoNQYxnMMYa'},
                provider_response_id='msg_01ADHu3dgMsjDVUWBzFpAfSV',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_shell_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        builtin_tools=[ShellTool()],
        instructions='Always use the shell tool. Keep responses brief.',
    )

    async with agent.run_stream('What is 2 + 2? Use python to calculate it.') as result:
        await result.get_output()
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2? Use python to calculate it.', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='shell',
                        args='{"command": "python3 -c \\"print(2 + 2)\\""}',
                        tool_call_id='srvtoolu_01UxJPPkQp71eQ8WTmHrfXff',
                        provider_name='anthropic',
                        provider_details={'server_tool_name': 'bash_code_execution'},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='shell',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '4\n',
                            'type': 'bash_code_execution_result',
                        },
                        tool_call_id='srvtoolu_01UxJPPkQp71eQ8WTmHrfXff',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='The result of **2 + 2 = 4**, as calculated by Python!'),
                ],
                usage=RequestUsage(
                    input_tokens=4627,
                    output_tokens=91,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 4627,
                        'output_tokens': 91,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CYvsKSFQ7yFXaDtpirUvu'},
                provider_response_id='msg_01CAcVpyAEzKZuodAeKRoBfp',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_server_tool_pass_history_to_another_provider(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    openai_model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    anthropic_model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(anthropic_model, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert result.output == snapshot('Today is November 19, 2025.')
    result = await agent.run('What day is tomorrow?', model=openai_model, message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Tomorrow is November 20, 2025.',
                        id='msg_0dcd74f01910b54500691e5596124081a087e8fa7b2ca19d5a',
                        provider_name='openai',
                    )
                ],
                usage=RequestUsage(input_tokens=329, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 11, 19, 23, 41, 8, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0dcd74f01910b54500691e5594957481a0ac36dde76eca939f',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_server_tool_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?', model=google_model)
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [[UserPromptPart], [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart]]
    )

    result = await agent.run('Multiplied by 12390', model=anthropic_model, message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
            [UserPromptPart],
            [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )


async def test_anthropic_empty_content_filtering(env: TestEnv):
    """Test the empty content filtering logic directly."""

    # Initialize model for all tests
    env.set('ANTHROPIC_API_KEY', 'test-key')
    model = AnthropicModel('claude-sonnet-4-5', provider='anthropic')

    # Test _map_message with empty string in user prompt
    messages_empty_string: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='')], kind='request', timestamp=IsDatetime()),
    ]
    _, anthropic_messages = await model._map_message(messages_empty_string, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot([])  # Empty content should be filtered out

    # Test _map_message with list containing empty strings in user prompt
    messages_mixed_content: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['', 'Hello', '', 'World'])], kind='request', timestamp=IsDatetime()
        ),
    ]
    _, anthropic_messages = await model._map_message(messages_mixed_content, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot(
        [{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}, {'text': 'World', 'type': 'text'}]}]
    )

    # Test _map_message with empty assistant response
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='You are helpful')], kind='request', timestamp=IsDatetime()),
        ModelResponse(parts=[TextPart(content='')], kind='response'),  # Empty response
        ModelRequest(parts=[UserPromptPart(content='Hello')], kind='request', timestamp=IsDatetime()),
    ]
    _, anthropic_messages = await model._map_message(messages, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    # The empty assistant message should be filtered out
    assert anthropic_messages == snapshot([{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}]}])

    # Test with only empty assistant parts
    messages_resp: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content=''), TextPart(content='')], kind='response'),
    ]
    _, anthropic_messages = await model._map_message(messages_resp, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert len(anthropic_messages) == 0  # No messages should be added


async def test_anthropic_tool_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01X9wcHKKAZD9tBC711xipPa')
                ],
                usage=RequestUsage(
                    input_tokens=445,
                    output_tokens=23,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 445,
                        'output_tokens': 23,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_012TXW181edhmR5JCsQRsBKx',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01X9wcHKKAZD9tBC711xipPa',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id='toolu_01LZABsgreMefH2Go8D5PQbW',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=497,
                    output_tokens=56,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 497,
                        'output_tokens': 56,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01K4Fzcf1bhiyLzHpwLdrefj',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='toolu_01LZABsgreMefH2Go8D5PQbW',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_text_output_function(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(
        'BASED ON THE RESULT, YOU ARE LOCATED IN MEXICO. THE LARGEST CITY IN MEXICO IS MEXICO CITY (CIUDAD DE MÉXICO), WHICH IS BOTH THE CAPITAL AND THE MOST POPULOUS CITY IN THE COUNTRY. WITH A POPULATION OF APPROXIMATELY 9.2 MILLION PEOPLE IN THE CITY PROPER AND OVER 21 MILLION PEOPLE IN ITS METROPOLITAN AREA, MEXICO CITY IS NOT ONLY THE LARGEST CITY IN MEXICO BUT ALSO ONE OF THE LARGEST CITIES IN THE WORLD.'
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="I'll help find the largest city in your country. Let me first check your country using the get_user_country tool."
                    ),
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01JJ8TequDsrEU2pv1QFRWAK'),
                ],
                usage=RequestUsage(
                    input_tokens=383,
                    output_tokens=65,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 383,
                        'output_tokens': 65,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01MsqUB7ZyhjGkvepS1tCXp3',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01JJ8TequDsrEU2pv1QFRWAK',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Based on the result, you are located in Mexico. The largest city in Mexico is Mexico City (Ciudad de México), which is both the capital and the most populous city in the country. With a population of approximately 9.2 million people in the city proper and over 21 million people in its metropolitan area, Mexico City is not only the largest city in Mexico but also one of the largest cities in the world.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=460,
                    output_tokens=91,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 460,
                        'output_tokens': 91,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0142umg4diSckrDtV9vAmmPL',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_prompted_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01ArHq5f2wxRpRF2PVQcKExM')
                ],
                usage=RequestUsage(
                    input_tokens=459,
                    output_tokens=38,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 459,
                        'output_tokens': 38,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_018YiNXULHGpoKoHkTt6GivG',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01ArHq5f2wxRpRF2PVQcKExM',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=510,
                    output_tokens=17,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 510,
                        'output_tokens': 17,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01WiRVmLhCrJbJZRqmAWKv3X',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_prompted_output_multiple(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=265,
                    output_tokens=31,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 265,
                        'output_tokens': 31,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01N2PwwVQo2aBtt6UFhMDtEX',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_output_tool_with_thinking(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel(
        'claude-sonnet-4-0',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000}),
    )

    agent = Agent(m, output_type=ToolOutput(int))

    with pytest.raises(
        UserError,
        match=re.escape(
            'Anthropic does not support thinking and output tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
        ),
    ):
        await agent.run('What is 3 + 3?')

    # Will default to prompted output
    agent = Agent(m, output_type=int)

    result = await agent.run('What is 3 + 3?')
    assert result.output == snapshot(6)


async def test_anthropic_tool_with_thinking(allow_model_requests: None, anthropic_api_key: str):
    """When using thinking with tool calls in Anthropic, we need to send the thinking part back to the provider.

    This tests the issue raised in https://github.com/pydantic/pydantic-ai/issues/2040.
    """
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, model_settings=settings)

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot("""\
Based on the information that you're from Mexico, the largest city in your country is **Mexico City** (Ciudad de México). \n\

Mexico City is not only the largest city in Mexico but also one of the largest metropolitan areas in the world. The city proper has a population of approximately 9.2 million people, while the greater Mexico City metropolitan area (which includes surrounding municipalities) has over 21 million inhabitants, making it one of the most populous urban agglomerations globally.

Mexico City serves as the country's capital and is the political, economic, and cultural center of Mexico.\
""")


async def test_anthropic_web_search_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing web search tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    content: list[BetaContentBlock] = []
    content.append(BetaTextBlock(text='Let me search for the current date.', type='text'))
    content.append(
        BetaServerToolUseBlock(
            id='server_tool_123',
            name='web_search',
            input={'query': 'current date today'},
            type='server_tool_use',
            caller=BetaDirectCaller(type='direct'),
        )
    )
    content.append(
        BetaWebSearchToolResultBlock(
            tool_use_id='server_tool_123',
            type='web_search_tool_result',
            content=[
                BetaWebSearchResultBlock(
                    title='Current Date and Time',
                    url='https://example.com/date',
                    type='web_search_result',
                    encrypted_content='dummy_encrypted_content',
                )
            ],
        ),
    )
    content.append(BetaTextBlock(text='Today is January 2, 2025.', type='text'))
    first_response = completion_message(
        content,
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The web search result showed that today is January 2, 2025.', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    # First run to get server tool history
    result = await agent.run('What day is today?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'web_search'
    assert server_tool_returns[0].tool_name == 'web_search'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the web search result?', message_history=result.all_messages())
    assert result2.output == 'The web search result showed that today is January 2, 2025.'


async def test_anthropic_code_execution_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing code execution tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    content: list[BetaContentBlock] = []
    content.append(BetaTextBlock(text='Let me calculate 2 + 2.', type='text'))
    content.append(
        BetaServerToolUseBlock(
            id='server_tool_456',
            name='code_execution',
            input={'code': 'print(2 + 2)'},
            type='server_tool_use',
            caller=BetaDirectCaller(type='direct'),
        )
    )
    content.append(
        BetaCodeExecutionToolResultBlock(
            tool_use_id='server_tool_456',
            type='code_execution_tool_result',
            content=BetaCodeExecutionResultBlock(
                content=[],
                return_code=0,
                stderr='',
                stdout='4\n',
                type='code_execution_result',
            ),
        ),
    )
    content.append(BetaTextBlock(text='The result is 4.', type='text'))
    first_response = completion_message(
        content,
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The code execution returned the result: 4', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # First run to get server tool history
    result = await agent.run('What is 2 + 2?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'code_execution'
    assert server_tool_returns[0].tool_name == 'code_execution'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the code execution result?', message_history=result.all_messages())
    assert result2.output == 'The code execution returned the result: 4'


async def test_anthropic_web_search_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Give me the top 3 news in the world today.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"q', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(args_delta='uery": "top', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' w', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='orld n', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='ew', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(args_delta='s today"}', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY'),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "top world news today"}',
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'World news - breaking news, video, headlines and opinion | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Breaking News, World News and Video from Al Jazeera',
                            'type': 'web_search_result',
                            'url': 'https://www.aljazeera.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '7 hours ago',
                            'title': 'NBC News - Breaking News & Top Stories - Latest World, US & Local News | NBC News',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcnews.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 hours ago',
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '14 hours ago',
                            'title': "World news: Latest news, breaking news, today's news stories from around the world, updated daily from CBS News",
                            'type': 'web_search_result',
                            'url': 'https://www.cbsnews.com/world/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'International News | Latest World News, Videos & Photos -ABC News - ABC News',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/International',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Google News',
                            'type': 'web_search_result',
                            'url': 'https://news.google.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 days ago',
                            'title': 'World News Headlines - US News and World Report',
                            'type': 'web_search_result',
                            'url': 'https://www.usnews.com/news/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 hours ago',
                            'title': 'Fox News - Breaking News Updates | Latest News Headlines | Photos & News Videos',
                            'type': 'web_search_result',
                            'url': 'https://www.foxnews.com/',
                        },
                    ],
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content='Let me search for more specific breaking'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' news stories to get clearer headlines.')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content='Let me search for more specific breaking news stories to get clearer headlines.'
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T', provider_name='anthropic'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='{"query', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='": "breaki', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='ng news ', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='headl', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='ines August ', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='14 2025', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "breaking news headlines August 14 2025"}',
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://edition.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'ABC News – Breaking News, Latest News and Videos',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'Newspaper headlines: Thursday, August 14, 2025 - Adomonline.com',
                            'type': 'web_search_result',
                            'url': 'https://www.adomonline.com/newspaper-headlines-thursday-august-14-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Global News - Breaking International News And Headlines | Inquirer.net',
                            'type': 'web_search_result',
                            'url': 'https://globalnation.inquirer.net',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News – The White House',
                            'type': 'web_search_result',
                            'url': 'https://www.whitehouse.gov/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Latest News: Top News, Breaking News, LIVE News Headlines from India & World | Business Standard',
                            'type': 'web_search_result',
                            'url': 'https://www.business-standard.com/latest-news',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '10 hours ago',
                            'title': 'Ukraine News Today: Breaking Updates & Live Coverage - August 14, 2025 from Kyiv Post',
                            'type': 'web_search_result',
                            'url': 'https://www.kyivpost.com/thread/58085',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': 'July 14, 2025',
                            'title': '5 things to know for July 14: Immigration, Gaza, Epstein files, Kentucky shooting, Texas flooding | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/2025/07/14/us/5-things-to-know-for-july-14-immigration-gaza-epstein-files-kentucky-shooting-texas-flooding',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Daily Show for July 14, 2025 | Democracy Now!',
                            'type': 'web_search_result',
                            'url': 'https://www.democracynow.org/shows/2025/7/14',
                        },
                    ],
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=5, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(
                index=5, delta=TextPartDelta(content_delta='d on the search results, I can identify the top')
            ),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' 3 major news stories from aroun')),
            PartDeltaEvent(
                index=5,
                delta=TextPartDelta(
                    content_delta="""\
d the world today (August 14, 2025):

## Top\
"""
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=TextPartDelta(
                    content_delta="""\
 3 World News Stories Today

**\
"""
                ),
            ),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='1. Trump-Putin Summit and Ukraine Crisis')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(
                index=6,
                delta=TextPartDelta(
                    content_delta='European leaders held a high-stakes meeting Wednesday with President Trump, Vice President Vance, Ukraine'
                ),
            ),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="'s Volodymyr Zel")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="enskyy and NATO's chief ahea")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="d of Friday's U.S.-")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta='Russia summit')),
            PartEndEvent(
                index=5,
                part=TextPart(
                    content="""\
Based on the search results, I can identify the top 3 major news stories from around the world today (August 14, 2025):

## Top 3 World News Stories Today

**1. Trump-Putin Summit and Ukraine Crisis**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=7, part=TextPart(content='. '), previous_part_kind='text'),
            PartDeltaEvent(
                index=8, delta=TextPartDelta(content_delta='The White House lowered its expectations surrounding')
            ),
            PartDeltaEvent(index=8, delta=TextPartDelta(content_delta=' the Trump-Putin summit on Friday')),
            PartEndEvent(index=7, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(index=9, part=TextPart(content='. '), previous_part_kind='text'),
            PartDeltaEvent(
                index=10,
                delta=TextPartDelta(content_delta='In a surprise move just days before the Trump-Putin summit'),
            ),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=', the White House swapped out pro')),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta="-EU PM Tusk for Poland's new president –")),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=" a political ally who once opposed Ukraine's")),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=' NATO and EU bids')),
            PartEndEvent(index=9, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=11,
                part=TextPart(
                    content="""\
.

**2. Trump's Federal Takeover of Washington D\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=11, delta=TextPartDelta(content_delta='.C.**')),
            PartDeltaEvent(index=11, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(
                index=12,
                delta=TextPartDelta(
                    content_delta="Federal law enforcement's presence in Washington, DC, continued to be felt Wednesday as President Donald Trump's tak"
                ),
            ),
            PartDeltaEvent(index=12, delta=TextPartDelta(content_delta="eover of the city's police entered its thir")),
            PartDeltaEvent(index=12, delta=TextPartDelta(content_delta='d night')),
            PartEndEvent(
                index=11,
                part=TextPart(
                    content="""\
.

**2. Trump's Federal Takeover of Washington D.C.**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=13,
                part=TextPart(content='. '),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=14,
                delta=TextPartDelta(
                    content_delta="National Guard troops arrived in Washington, D.C., following President Trump's deployment an"
                ),
            ),
            PartDeltaEvent(
                index=14, delta=TextPartDelta(content_delta='d federalization of local police to crack down on crime')
            ),
            PartDeltaEvent(index=14, delta=TextPartDelta(content_delta=" in the nation's capital")),
            PartEndEvent(index=13, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(index=15, part=TextPart(content='. '), previous_part_kind='text'),
            PartDeltaEvent(
                index=16,
                delta=TextPartDelta(content_delta='Over 100 arrests made as National Guard rolls into DC under'),
            ),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=" Trump's federal takeover")),
            PartEndEvent(index=15, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=17,
                part=TextPart(
                    content="""\
.

**3. Air\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=17, delta=TextPartDelta(content_delta=' Canada Flight Disruption')),
            PartDeltaEvent(index=17, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(
                index=18,
                delta=TextPartDelta(
                    content_delta='Air Canada plans to lock out its flight attendants and cancel all flights starting this weekend'
                ),
            ),
            PartEndEvent(
                index=17,
                part=TextPart(
                    content="""\
.

**3. Air Canada Flight Disruption**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=19, part=TextPart(content='. '), previous_part_kind='text'),
            PartDeltaEvent(
                index=20,
                delta=TextPartDelta(
                    content_delta='Air Canada says it will begin cancelling flights starting Thursday to allow an orderly shutdown of operations'
                ),
            ),
            PartDeltaEvent(
                index=20,
                delta=TextPartDelta(
                    content_delta=" with a complete cessation of flights for the country's largest airline by"
                ),
            ),
            PartDeltaEvent(
                index=20, delta=TextPartDelta(content_delta=' Saturday as it faces a potential work stoppage by')
            ),
            PartDeltaEvent(index=20, delta=TextPartDelta(content_delta=' its flight attendants')),
            PartEndEvent(index=19, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=21,
                part=TextPart(
                    content="""\
.

These stories represent major international diplomatic developments, significant domestic policy\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=21, delta=TextPartDelta(content_delta=' changes in the US, and major transportation')),
            PartDeltaEvent(index=21, delta=TextPartDelta(content_delta=' disruptions affecting North America.')),
            PartEndEvent(
                index=21,
                part=TextPart(
                    content="""\
.

These stories represent major international diplomatic developments, significant domestic policy changes in the US, and major transportation disruptions affecting North America.\
"""
                ),
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "top world news today"}',
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': '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',
                            'page_age': '4 hours ago',
                            'title': 'World news - breaking news, video, headlines and opinion | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/world',
                        },
                        {
                            'encrypted_content': 'EqkCCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDJG4eFuXHMyOtzU+gBoMHKBRjV1QEadg6zlbIjCZLJpWU12fOFX5D8hg52fTdxlJYaqbZMhrhPjKAAuwxhkMOPYFyaUk7XM3mcUAzdAqrAEwwPQ3fLoZhOecxIz5vbDar6Zsokam64tDTNCfTQLRBdV15DkyJ2d5sixzIoQTACBh6hgluv65odozld+evav6lY/9/XUQD4RTWdccAvzmj57FwBJ7I2cZeyxCy2SvSFLA9N6XoM+/iycTNaEtwWYhW80ksf871AcZEzv4gJRYnZoQvD9Qufi6bUL1wwOYB5eOUCxWbtnElsrakgxPaYcvkhuLPgNSQBmG6XO4GAM=',
                            'page_age': '1 hour ago',
                            'title': 'Breaking News, World News and Video from Al Jazeera',
                            'type': 'web_search_result',
                            'url': 'https://www.aljazeera.com/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '1 hour ago',
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': 'EvkICioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDDQfuprKqbAtk20SBxoMcgQ8NRE6fxZzwrcNIjD1Y6oQUAuFBWJl/CbwbUP38K0WNKl0RxxWzta+Ae3zSvXS5TjN4HdYSpCswRvDQ0Yq/AcVNFU9giafOUTlMW7CV1GLkau9rrB4aNSYnQl9CU3tBFusN6wYtF8znSWYwwxAFkKuhD29utoaH9J1lAuKdJkQM98t0Hbx+xpclYfb0KoVsPl/8v7HPtLp22PqJJp/Vh5dfWrkm41OOBGa5oG9Y3CD2MzUeM4DuuCFDDVN+gu999McvfjBJXenWmLUt+yqk+5CEioHiF3mqdzM7hf/mGBHi7TfeSl2+Ad3v/lOE6sMrDna6X+Gzsxu4gSWL4HttmPJQHcRxRRm5YGEsr34hvWerL5u6SaWos3lxamrReXe2Vg5GxouenfjufJpO/4ZPBxufHJ/LOav03Rcdf5mxw11uRsaJUF9EVUESmmxhfR+uY8Cb6Qe8jwZm8Os41/Vqif8BSXrHZ1a8dNVhAVymJQ80GLSJR2OCH1ri2BlHhosaTzL5y2+502b+Lebv2rJtIPdt4Fhxubz+rje2BdwF5AY/8G7pm+qqHhsrG/MOZSLIYe9I8rmOe7J+kvzMcOJm38kaE3kxXNf5CPEyCW8lfijwXeuaSYGesVdulSaoChsQbfIqbjSTjSkusUd5x7YCSLS+0lVMj6agp/DTGbG4WuvdyScOkOpJkg/Rc51UcC10B0Qwa+XfvLWikrEGQIrqa2zEbIEu6xYMn8F85zqOslX5RL16d0FDMmNJbh/cR3IOIwCvD06VWrBSOIqNOtDs3XKIaDqVVZKu9mkaWq4AX7mEWlyKd9OwHwBDPik7D3bWtK8xnQb0G/AGK8mlDgBmtMCYL5yna1xAYEhcaIF9LVJUzKBaqFoymvLAvRLzBFmRvDHE5cIPTh2OphmTrEijr1FD3kmczBDTwXTENM0eJ1jcB8UDWk9oGXQZzwhg4w9yrW+PBcfZaSB4q9PA+A+2F8vhIExkrPli+WschOzGABlVKCACBIxte3UiQ+D6FoReel6h0uouDf8bS+J62esIgHCpGbwV57dYEc8maOtGvHVm+R224mev5id2OnwaqhqrZ2MNFt7J/Dr+n0tWa/KIDnN2bzn7H2YTSxtGuQG1jbNkZgeS+I4yBtEN4rNPO+dFBO/KUGP+ehYbdM2wG0tes/Doduk/geCt9vFYmpiBGNUZusQ6YHlWVOiYatk3I+EeTBWifjTa4JgGMMOmJI6mibHG5AKEs2qk2cOTf7q8aUv/H0D6mul1/TENTo3KwwCAtgg+9+Z2couuyGCHNrTp2sXwlPG5zaL6Hs44zx7j/Fn3FnmbIMRWHTTtD21TG1D8g01ODNo0R9Tfpx5jtKQJsveiCMhV0xVH8EQclxADlpyqEQjQhrXGmfE7e/9m5QTCDfwrqaCThIKutNsT0jWYDXE1xxeN1mV524IymwYAw==',
                            'page_age': '7 hours ago',
                            'title': 'NBC News - Breaking News & Top Stories - Latest World, US & Local News | NBC News',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcnews.com/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '3 hours ago',
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '14 hours ago',
                            'title': "World news: Latest news, breaking news, today's news stories from around the world, updated daily from CBS News",
                            'type': 'web_search_result',
                            'url': 'https://www.cbsnews.com/world/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '4 hours ago',
                            'title': 'International News | Latest World News, Videos & Photos -ABC News - ABC News',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/International',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '1 hour ago',
                            'title': 'Google News',
                            'type': 'web_search_result',
                            'url': 'https://news.google.com/',
                        },
                        {
                            'encrypted_content': 'EsQHCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDOlbTIZR7LE+ht640xoMO6oGh2/5jXtKmt2PIjDgd425JPaAMCzlqe4UoZXj9uAw25vy948R1Ml1K/b77P5epdCbEf+afr9PFoQXlIwqxwaRq+tSfJGZEi4iyMpYT1fVkqSXknqpDfLmlkyPJAXbZxzGNbsLaUDTuZ4gzwMk4M2/ypCXThqBO61NbClsbr4OKPQumWcYhRxGFjWo4PT9N0kZZcq55Fz0e2zfGRtt42kGXJa8rFEWzafLgocJN1ghuEjT1XsdpDo38p4+UjxIXNJzDSHiC2c4yrNOXQHYEpLg6uWjw6VgvUfltVSacwvZ4G7+yO17Curi5v0yuDzo5aYnQBlDgHyaMTvypDzaa7wWOmfho8FiI1LVUSMb0mHLuR86sJGTkpxKvShA4xe4NGLGXA/AKOKtSrHNd3AZ2Idpu1S3ot8Ki1vs3KASmV023ZNhEAsY5apWt++6tFMJ3vdJUywLs/R0AhMcd2GJrzRscKSaz3hjs9BlOY2UliUJUvchjMzo7IU3gr9ThCIGj7sXAU2DXMLzxdgKZqRt3JscVPfR0ivtq7IOgI1+hqui3p0QRS0FYPXALvXtF46+UfRP8bq/kbHPA9NbO2EKO8ffx0Z2+wfsBMJqy3IkBt/qsJdM9M2WQaTvtnDI7NKfs7UxwvqiuI4+g/yzTZuI1NEN9orlWkrYhUppi3flwopbIQc4sVQhTj5ZEyPh+bdoNYI2FyguWDJptCYtjdv5Ao5NPGVnET9rwLzFH/GzMWixY4UvZHzshnGRWsus0dGy5fSM/6QgYLgo2LqriFd97xQ0RbqnbNL/Id8epv3Nsdn5aewvvdgWRCfe8KV+DHhRzYxyLL7hFWE6yQ35fgmTj6gyCjfImC+xmdNW7Q4SE9uWfDMHLl8LB88n9e42Le9TNfyaEvKWVqB2JJNLm+XJki2vxpW2MBV6bmrmYL+5sTMoSOy84xmrGJjRLQhXrPxlACVlEZSGN+ZpOC+SINgjRapVfVd7U/U0Ckc+42wG/rctypHw2i/WsEYPPO8lvo8jFO8QBMDw17DYLimGLW6av7a6bXPc4xi4J1t6l7bCrHA4wxpr+knwazPENHExNleypISpfHkxLEAo4IFBmYLaJcP8128e88mLtVs6Pr6Es6oUwVd3oPmLeHM/IHtpwSTe7NFvz32UgeMK+Ob8JCgUEeMWMbbyA7iJAmEOSx4vrxZVTXyz8xfrdRgD',
                            'page_age': '2 days ago',
                            'title': 'World News Headlines - US News and World Report',
                            'type': 'web_search_result',
                            'url': 'https://www.usnews.com/news/world',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '2 hours ago',
                            'title': 'Fox News - Breaking News Updates | Latest News Headlines | Photos & News Videos',
                            'type': 'web_search_result',
                            'url': 'https://www.foxnews.com/',
                        },
                    ],
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
            BuiltinToolCallEvent(
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "breaking news headlines August 14 2025"}',
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://edition.cnn.com/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': 'EqkHCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDPjy4Tk2ZA4s5aAgVRoMg5FaArA68F15P7i8IjCZCpv40Dn+y0KLe3x/ljplOd/KcfdBTQvNmuUxny/UgY+KU9TM7COzrcwuUNw1uKYqrAbuTehNwNzgSjtez445VA7hf6k2fwMw4y81HsGAB9xqKRvV2tcvdObPOMif9xBZVXnvgzGOAaXHl5PZmv29eNRlfCMFYfMAaN/rdvLDKzhpC/IDd/Y3YZEuWfx4BiXoyJor8I9s+VQ5SjFLRfiLEU9H7l/2Mm7wNf8gWb6+X3gB5Fkysf5FLD6gxXCP21G/JdGMWv/FmcV4yQ1+lR0LMz7rNETtO2p8BAZgrZf89EEovpFcE1PNbrmAn8zjf/EWYHolVLFMdYQ+VBRqPQOjoMhHbQ0GGB6FioxFFodvXPY0XydttssInrn8D//87b7KAQXdfpjOjfb16gczZDkOEx4frn4SokkvuEQZWnlVtQmz8sbCSu8B0xBH0VdPXUGOFYh/GKOz8oiL5h2d2/Tqgh9C2IXNoFkxfYHlUQPaBah5Hfb7u8cJ6FpdR8Kd8YkeFQr+mGyGLEu3x4YdpqJvp9V3xWd2jucxP5USw8oHBncR/HFXotoXzHjyy3d3C+Lac1QH+x1/3gafKud3PmncfSwzaPrjcGpClkrokrsXZSCubdx55KQJCW8b1NEKp10w5xuhSg4aYH64pnEcx0IOTU620vb6k6OBMhUJ7wECtbyh/+I1xL7FVde5hNsyH18jsO2DwDUdleGlf+n/pCfTUlRo7ULt9I7Qq3iIo5UA79bvIOlwRYH4vfNnKEKKcoY2L8fhrsHrm1kaAYvPt4z7DdYHy/pd8Oixizos8O3zB094KAG+A6j2jjvUKhM3lVtPRGRa/eGYMSkkNW7Ik2vnMoPkvRsm0LM3/KNHOV+7SNvpM+/gW41rU/XDa55Tcmnw0uJsJOpOXxGhxF1MTXslCHLoE6VzuQnww/KAx82ZHAL1DVVxG9Cnk2QEH9GIx5IlK1og7OCoWfYvhV5ivlrazGDaJhx7WilJaX9TY5KN1dPRdxwvfl/5jJtRa8PqYsnk/oXak4Do5qCk6hC7uec9zVuG38j4+6uUeIB4AcBiueDNT8GRx9ccQ01v06JJtIXiXRtZQ2lCmC5ZHi1a0sA2kfCHCF6mu1VbopBV5vv00XQ1xvDFMIX1dBVsxhy6nRgD',
                            'page_age': None,
                            'title': 'ABC News – Breaking News, Latest News and Videos',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/',
                        },
                        {
                            'encrypted_content': 'ErkNCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDH3KhBxmoAj/cAFVWhoMDhMOdEaRQtsF8KvrIjAD4/yraLwnP3qTWcjkT0EyUO2T4OF+8W8AauYBc03zwOdA/Gx23/UZQJY/1etRBuYqvAxk6yvK+pWQSpNneXso4n9Rfm4ZL5+EaFxiLrGqPX/aeJnvH1S0+UMZwKITAxJt0wUJZ9XXkRNGmuNNQmnZ2MlBOHv8RNFLP6LYHvZuRdqWMV/Yr8XwVgHYnQB1qnjHkhS01vD7QxSuNh6JNJrRoV7eow34hxGYGNRFHbySXS8TMGrwBSdkYEOlOd22Y4EGF9Aqidv+xmATmuxScbZxVV5VGA14JjBmMHizr6qeSqW5gV5D1ulkn6Bg2thLtJCO6mQEFUBfZH5TwTCkGzR0Mo02cRrB4RAJuK7orl0VKFFcMs/lhdGs/Rhg6Q3LD3O9zsFV4Rx9JtixLH9XhCxOu2MWxZqIBUNu9M2x3BWQs3rlds3/4x7nVpLe8VJV1cJ7DA+8vIglA42xvbCuLUtUDQpUXDDod0XUIqoQALcrSsq6QOfO5nxZwBr96EbREtLcWEhoM7ymsv80FzDk4ftHz/H2glV2qvc1ql6fomovj++OZZJqa9COXiQ9itTvUYo+lBwc7Wt4JjjUUjbDTHXJkhcIPnKhuMnXWxput2UySJXfAEvwSOpddlK08IsoQt/meJjJVU70hvwMXKyn0KCkSCdQEd6GqQhlAjfWeoD0E+WEFsEXyecAXnXnBReFRSZ28jY2eMiMhshdE3uK03q9xvV2JH+I6O1zsROLcPCqLw4+9TwfD8cjgXWmElRisFQ1tPuv926uODgpijMEPzGoONGkSF3AKusabif7S/rQhXSA9oziVP9Slqk9eepeXbHb0dFXqpLO0iLRvZGEGimzd5nRYbP4//9u7jMa4USbbC+rPR3Mz+pP+7GC1ke6HxmZWzpqYvIfHTSmsppdxAvUJreOf9CzuR5e0fZZ//N8fSIneGn6eMWQ999wlIHIOqTVPARc56013cXNWnNKMYVi2rx/bBhBWwtp34yNhJvnZN6P0pcmCA0wpotu5mLTxEw7MTVPSNotot8eOt21waNFGxf4NTko6Rs8Ud5sV/jfJPdj3bBuQawRdCUyvKjA5WYH/Yeb3ElDCIlRCLM9geP09BgSdGRpdg1wTgZKBLkxcJ+UJAnONIUqwomRENYSNhfDT2EzvNMK9m/NT0wfdlqSwqgMR0Wleq5e/0me0UCFtZguTbRep/UYPeLOyMED8vxnCd3amvxsQAEJbF2dYXsZZournIRQaMy1Lv8MvW1Z2T/wB/YH7k/5AYeTK548rqnxWDQ8slH+4IcpE+AXpAIBYppLIA0Ol5qdb8hgxifViFZISPA7AVhemLdZ9DGDyi2s2fTZzMyE970jZPx59Poh4KYVtcCJcfskomQjrtFaRgb6ESoJyLEbbhnnXA3/MHcp6J3w7MHR+VLliGozu9NZZQXtaiHTxMv7dtyA+PRRQ89rV7n3cIs+LO0mUtVc3JonSJki9xGnPfNUO5DK9IBS8ftWqDoGS10eylxb/qe6G6EfVrbfEhx/IprMANybAdu6n+v8H1LbSmQeWG0YRlR/A7Tar7SJzDYoDWDasjPVlb2LWE9DvLhIaxTwskMO/XTzEkUpimElGSNTyZBfo+EBpahCfHyBgwPq40IPMspA939S/1oE8N8mdNM2ijQqpz6nspiwffFT8nrPzFPrisIK75KdKh8a/2BkzCM1GToqWHMIO4KxiWYgnAoYARpvxim7M3mwHSjJWPwvtq/yb1ynDt1tQ+5eFWPN6b4j36SH2jXn7X6JWs8ub3MR18NeVDENL1jERts0PTCgGL5ZQYKBG8jJTxTLaUJq0X4qDlmF5k6O2pb8fz+RZ2e4oNeeSj+UJ3ULWZjh6O+kaEUxbn19k+ONLPk5J0bux0v6Ldg5KkP42V75ZEj1cKcgRGZJWc8Ctkx12yymplyTBYfYS84Jd9CGhzvsUaR+MOeALZDkuqmQfHT6U2Huh+U9lQHOcWV1mZzU3Zu7i+pEgSdI7PRzaODiorEp6gc/OilrHfX9cEXxN3MCxTHTHst5El/mu5uoyO4GI9qEpAfi+xReBxbLB0IUUEdad5esgxUZgqQ2kaOmwRl14F1LiukZZJm5du3TDsC5dFJB2d8fdUQHpHYrjUo+IYdNFtNNrTz0J2OpVD1LJtH/umUFcq6MAIaUJT27zT7DQUYjtNz1m1obX2kYAw==',
                            'page_age': '4 hours ago',
                            'title': 'Newspaper headlines: Thursday, August 14, 2025 - Adomonline.com',
                            'type': 'web_search_result',
                            'url': 'https://www.adomonline.com/newspaper-headlines-thursday-august-14-2025/',
                        },
                        {
                            'encrypted_content': 'EuAICioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDGUBq/AWHK+eN9ikrBoMDCe+fiGygXm0v8vnIjA/d6J5VILrWucTWhGFWr1ZrdnjHbnwjptEV0I2qYQc2aIizW3/jgDWUp1cRzZpcLkq4we2pNaORWVpGV7eKefzCX/BaTzjERAu8eCilTc9y9F+nvenlHUe9mtdCp8QFlw4gS5Xp3UoDhs7sdU9qZE/b76TiE3ay1jQ9qEJ4tPnU/ZHGCK1x0Rxqr8xUlzghehBzM//+fqcSMo6X3D32eYiZrxhnEpCPyavPUDFiV7xYsp86b8FHeLRNH9+dP8UnvMLjf5Y78prsGp6rmCG5Ri0xb/OSmPxdnbziPqevizibu+bng4xep7Mh5DdKzrTHXZytjtrlyqb6ELJDgV46aqjLAzrbPL2kfXj8m9N89Pjz1YQCUbqfdRe8jOcgTuWQj29VmcIicrE11enFD+yQPelDXo5vMy01Zj6PaiTHv31+0KCUnPm/FbcB4Vl4BMrgp64Cg9d26GYyZI5l8YCoRycwRIP+PmBIAFWeP175EOlvFMXP3oaiNjy1L4DbeqjE4M3uhBIJfG/PGA/XVl3UDK+GwOzCuy3dcpI434upmL3oY4mxgZbbLIr5x6Rjvva+hazC8lS8haiCQMxusAJccrmFlh2oaj3RiKugJxFfkQfh+zu56IfcadoYunl9EgtVEUNl2CrXx2lHajZMKzJXBuHUYhXtxc1UKlqGrG21KilvTyxxUb1jyivdXG8BfxLDFLbinxUo4Bo840Ip3Arop+dDkAADM5r6HtCvISJ/3gkRh5tOzhaq3z087gWCQ9YFT2iqWvaJUSBOlRObK1ZGFsaIL7L9xuLtKiOip0brk9tzQKhghCEIW/hqAeZfpmvcLkVLlC0TZ8YHtIy3il2X8keUSOQEQnVsllKu1o3Qvm88kVPUuM3XVRbqWZ34IWCB5lNykt/UxNSwj0CdP6c6lB0XKZvDHTnw71SDAVup1sVEPhKJQ25DuuR6ExUpkrS/N00TQi5v9TBhKHnjcJwDD2nS1529IYCfBqTsAaKIJfVNB/26NGyLIpMnNUo54KuVyqGN61DQ/7PXolG4RCSNnk6iktHnEBpCebNzG4WnWfQeDdnkoLPvSVrQditQiGCrECHIMALs1r27mKpiZ8Nu1nJWkTkosBptMymAe/MpJxOgI0zGSq8Fv5LwchwCUFkECB4cqW4M8vo7IfTbd+dX41ZpkdL74NBeOl8zgigQoAzVw7xrLFlfz9Y22WnIAZlDjizMwwV33xLZK9fPtTszbeu2PJ/FqFDDRhasbcNIbqHKQyMgZS41B0arlM1cGEJrMjKYjHedHtnhxP2sDuy/xhAG98AXBe7mA+ffofonOAXlMv5Db9zXYTy2s1DZJjCQpyq2m99tl0s08tqUUJ1yxjXJ1L0mjsVFuQ4Qdd2KIXZg2KM33ce3RgD',
                            'page_age': None,
                            'title': 'Global News - Breaking International News And Headlines | Inquirer.net',
                            'type': 'web_search_result',
                            'url': 'https://globalnation.inquirer.net',
                        },
                        {
                            'encrypted_content': 'Et4ECioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDFIPj42woU98JMQRhRoMOYuBjzQ62lFOxkn/IjD9AaavhYfhuVUR+6XA55O0rVbWwPlUZjgIAw6N1kXBv1Y8tQM1FE/uKJSLLR4ixN4q4QMIQdUoYsT+zyb16Ndd4qa2I96kPDnhBy4rXGenBrnwzaBiApJ90u3rB6aJgqwx+zISZ9Co+YB2KUrNtvdoS9YMmgA1qxw2BWltYVXsvToBjJWmqyyQylJbqXoVF+v7sqwcibbPipASR7JB+xjVKD/APg6dQV3CjCFCcQupR8bMjv4IRk6ka3WmxR7fn4yurYteWdyBPDAN2dzKS0Bg+z4KwUGsYikVXSKO9/qJJKsXKiZfxWw8JWDmYJIAqaVody+KON7XcATd9DIdBXSwhqyotmqy1csVklrT6pWrptr6PjBGcUHKMP64tcVtS+JJ2w3+uwIl7YKyuwU89CSE4BbZ98SPGkQZKn0s2t939eEx38hCPJ+PG/R0Mka9+wy7tAemacDc9JrcmtcUVri0d+xxultUgeQtlLTyloW7Ufz5FcngQC4RduJYC8K0iuLIsvXWGW/c4HWXqtXE6cMgSuEzsIZxLNpZG1QD1W5DEOuqQw6LtSoxWt+9VncSsGfIBT/RMsBm2knlLzmNw6iZ/HvKv1IHFVWKKyda3F94XN3Pzqc776uQTOxxwHblQPcyTQ/AewMkfhkvuFELT4ZaYv0vzB9W8qh/dZyczuUvNeQc8/ISVZcHF7ZaTaf5aT3nNpk/GAM=',
                            'page_age': None,
                            'title': 'News – The White House',
                            'type': 'web_search_result',
                            'url': 'https://www.whitehouse.gov/news/',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': '1 hour ago',
                            'title': 'Latest News: Top News, Breaking News, LIVE News Headlines from India & World | Business Standard',
                            'type': 'web_search_result',
                            'url': 'https://www.business-standard.com/latest-news',
                        },
                        {
                            'encrypted_content': 'EpcJCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDJ6YEvrSK3v0CflI2xoMtLnlku0QNzC9V2sNIjAaCI8ZW3hnESOeFLgxg7xvhItv2BhoMaKQLmXFzh9Fj9EpUmnjNAeTw5PFiomhzCAqmgjgX7oo6JF0it2mfnkQXGE6w0vwpo1J8BlHDzNkimQhyWbIq9MF4zMkXckvcn67lZbvdZKYzTeOTALJkqnabeXCRQGGB0xMQsQznbmgz8Es3j4UAusK+vvgi0Ou0sRQMFliQeClYeA+TFyiY+fHayamH6zdyNZEL61FLRShDNwVwpjjhFgAQ7x2rOUA80Xa7/UkBmMmeqRvROz7ehZRTexsEkfNtQXQL3mBPZljNv+8QrX8zudDpscp0sa7PfRotycsZw/qdm9iGzshJv85A4gPHQdKPGxt7tEPeEsBgjQ47RgUqOsyiTi1mubzk9wV9iOv29/UN1ewegv6fWSjJVQuqa3lCu40O57T69fA5A1JH4gbvTXuwkoaPcqgjhUmEnmoBDFednqZvKRTRk+Vlk7Yx5xnHnrmlBOiwE+5POi2t5ZiNlZkWxvPFn8+PQPv6Tz/fJcTVes3ERf0dxxRMuhFNqme1LbOzMd6jy+Vpu6OnrHiZ2kVKV7XPg4MKRMGx71pioPVIN2lZbZDWlNrCT34uFLcw0A9KyQJ1dQpdzVjktsLgNvK6PX6o6NL9cfF8z0vZF1LGNSDr6ZqvrJzJOFhzKAujU7ZsdXfUwbOT8CkavTSvYgKEUJLHXzHsW0ehWxgsi6/v5OhHgRv+bt8kp5JV9b9PmfZk3clzi5my1I8uP0NSGAdgxIzO6+wruPYr579w4j+JA8veck/ODwgoPDZTwgbFK/nQZZQAl57BCDMSYUSVRc0TJjaV/u0YT8Xs35X3MUADISZIO6d1Cg9vLkERjSZyc1DmP9pTMR7OY9D+p2bFnqd5Uct2e3Q2Gi3YKceeiuRYc+gDgJXXkAX8rHIWid8O+VnFR/mlyvA3xGeh9Y5EssYZvGHMz6f55+dS7ghdczYbF5Xo4bsnH7SqPx23jdOVymLx2iqRKN0Q794ImwrNywyZeJZ++cfdba0IAXvV5mFI36bAcKRbsGImjBtyZxkx0/Ywano0rflkkTt1o/03o7hHy8TLmpkn+vFHOhCRZajkPqFqDt0zcmW/YJdAUgkRZEr3MqXQ9cMyWhDvOH5fEYaD0lE5O2c89UwpmKPPpx0UTrptm1k1KfVwnRMdgoGcB6dJ9BFze5kK2si2lHvWMpKvb1Gkn0rLoVQBB2fDtIh+eSUWapX/gWHGpe2MxNZrbvPJVeVq7MiCsc+VhsqBeRg3ICyfQw06Msac7T/HYROMlftyhErs63avegpFcEUMwMdk51gBE8yFjxOU4SCatbj0Cu6DjcyaXF+wHY6B50kDwnTPfrB8/TxF4ySKvhgEsMRpfDkhJomytj61mT/QYPIhBDYpftIFLenxwm+AOYcUT8pJ3nWmQXJtevkQ4EtIuX1FhsJJdRDnb9M7HgNKNpXGZhUOi8YAw==',
                            'page_age': '10 hours ago',
                            'title': 'Ukraine News Today: Breaking Updates & Live Coverage - August 14, 2025 from Kyiv Post',
                            'type': 'web_search_result',
                            'url': 'https://www.kyivpost.com/thread/58085',
                        },
                        {
                            'encrypted_content': 'EsAICioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDNSMkvUZw/5yzrXdUBoMvkJANnFVb1oCSduIIjCuTLGU5Rs6yHsUew+luIpPY2ULc4kpC5BcyW2SzRbQVRz7tyCdTeA4LcFub26oSBcqwwc8nTu21JMx9Vab54UdwsRUtkitcl0iiLfbFLao/NuHT0ttduqwzaqYX+nek01CV7W75AYRNtjgCdnsmcNYkzJLXbAhH3LK+hdvos9j8aaHX425S1tqZItis6A1PlW42luWgZowledZ1pKpfub8icDPvPbkdbkqSAcI9dt9p+InSRjIeusldfJpYFQReYt2W6+HOQNeKpuYukb2Z3u3F8LjnvGrmEIRmVi0mLp1hQRhTKm/JhMAfJjFzaFWY1SFt51QI2EHUNUHKXdTocPuMpGE8Ap+Mwm/120GJUa/Huh52LcbJusZbKROvuUOvWO1WbkZGmpn1HzKgggdrzXlj5dr/wAvml7QxkzSq4IN2sxPXUYMs4U8vFeuoSIjU3WwkDriEafqAwR9qy/IipsSqLg35wkjz7YOXU3E6lwVSfYGgdMH+QGLu/RbGd4YzsaGLD7XCD/3rXORNDKyffEMYNXTnvNTz7BwyCJJdumFt5tAbLEVSHiskk2R64EtRFiqMOCzVT94jBQTXm8B0K0oB0TmAy0sAlGNGN7QyWOkmTNWY8TjP44d1qedmDVwmpI2ooNVSb8Q7s8cxn6WX0LoJ7EPL3YdCW8qbJvmFb6xxVoZIp0UcJ+sBBCI20/LSglSJ4KqwU3aDrS7PLYPgTh8Ajqgiudid95jhjvCrUnJvIK3hol+FIlB8Yze5U/D9EUnv9p60D1czexBKTLhdT95ELi+G2q2GDQEhD2kuVR4SZtvtU1sNZwtq3ir4aF3ZKO8jJJsv3maraefZLy5fnDd/Orryx6TcFR7UvjWBRddCV5bAj1LIo/6rlkWChCUJ0lpUljuHqPFUqYzE5rixAb7jEh1WpTPjMFk3uW02AFWKKVGFEb1ALmS/yggECktiRBz9LFzfbObLR/p3q6BMRnCKvjWRe36W/7qCs2+5QFR2YUEBCuONxSfzd1lEASbQ7dO9HrZhlvVnxTWNlhSYWU2NNNoENbg2CnEt6zSRyWoI513WzEwE52pRZnkEiSloIEwXMmjWxTrsvO8PmpkCZujaHR1b1iNIe6I3UrF3dqKoe01/tXp4YKKjUL8Be8vd/2TPV6pHLfeFJFPnmuP3BUqj+Z7zA13Fy1WxTxBai/mLToZCbNhig1dbQtdIDL52pruhs0KEHTbxpcsQeqXKgGgKCyRoeb9AL+Z9u5DJRCq1nkampFfnB82U3fWyuV1vCNxQZ9yskVp1pmVNSACElKHx8+KtJXNuafLTzQR7bosR244kwCghoAJzKTl242UwUPj+00ZmWMYAw==',
                            'page_age': 'July 14, 2025',
                            'title': '5 things to know for July 14: Immigration, Gaza, Epstein files, Kentucky shooting, Texas flooding | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/2025/07/14/us/5-things-to-know-for-july-14-immigration-gaza-epstein-files-kentucky-shooting-texas-flooding',
                        },
                        {
                            'encrypted_content': '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',
                            'page_age': None,
                            'title': 'Daily Show for July 14, 2025 | Democracy Now!',
                            'type': 'web_search_result',
                            'url': 'https://www.democracynow.org/shows/2025/7/14',
                        },
                    ],
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


async def test_anthropic_text_parts_ahead_of_built_in_tool_call(allow_model_requests: None, anthropic_api_key: str):
    # Verify that text parts ahead of the built-in tool call are not included in the output

    anthropic_model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(anthropic_model, builtin_tools=[WebSearchTool()], instructions='Be very concise.')

    result = await agent.run('Briefly mention 1 event that happened today in history?')
    assert result.output == snapshot("""\
Here's one significant historical event that occurred on September 17:

In 1939, Finnish runner Taisto Mäki made history by becoming the first person to run 10,000 meters in less than 30 minutes, completing the distance in 29 minutes and 52 seconds.\
""")

    async with agent.run_stream('Briefly mention 1 event that happened tomorrow in history?') as result:
        chunks = [c async for c in result.stream_output(debounce_by=None)]
        assert chunks == snapshot(
            [
                'Let',
                'Let me search for a significant',
                'Let me search for a significant historical event that occurred on',
                'Let me search for a significant historical event that occurred on September 18th.',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                'Here',
                "Here's one notable historical event that occurred on September",
                "Here's one notable historical event that occurred on September 18th: ",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marke",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally.",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally.",
            ]
        )

    assert await result.get_output() == snapshot(
        "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally."
    )

    async with agent.run_stream('Briefly mention 1 event that happened yesterday in history?') as result:
        chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert chunks == snapshot(
            [
                'Let',
                'Let me search for a historical',
                'Let me search for a historical event that occurred on September',
                "Let me search for a historical event that occurred on September 16th (yesterday's date since",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17,",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Base\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), \
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, \
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, an\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristoc\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat.\
""",
            ]
        )

    assert await result.get_output() == snapshot(
        "Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat."
    )

    async with agent.run_stream(
        'Briefly mention 1 event that happened the day after tomorrow in history?'
    ) as result:  # pragma: lax no cover
        chunks = [c async for c in result.stream_text(debounce_by=None, delta=True)]  # pragma: lax no cover
        assert chunks == snapshot(
            [
                'Let',
                ' me search for historical',
                ' events that occurred on',
                ' September 19th.',
                """\


""",
                'Here',
                "'s one significant historical event that occurred on September",
                ' 19th: ',
                'New Zealand made history by becoming the first self-governing nation to grant women the right',
                ' to vote in national elections. It',
                ' would take 27 more',
                ' years before American women gained the',
                ' same right.',
            ]
        )

    assert await result.get_output() == snapshot(
        "Here's one significant historical event that occurred on September 19th: New Zealand made history by becoming the first self-governing nation to grant women the right to vote in national elections. It would take 27 more years before American women gained the same right."
    )


async def test_anthropic_memory_tool(allow_model_requests: None, anthropic_api_key: str):
    anthropic_model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'context-1m-2025-08-07'}),
    )
    agent = Agent(anthropic_model, builtin_tools=[MemoryTool()])

    with pytest.raises(UserError, match="Built-in `MemoryTool` requires a 'memory' tool to be defined."):
        await agent.run('Where do I live?')

    class FakeMemoryTool(BetaAbstractMemoryTool):
        def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
            return 'The user lives in Mexico City.'

        def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
            return f'File created successfully at {command.path}'  # pragma: no cover

        def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
            return f'File {command.path} has been edited'  # pragma: no cover

        def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
            return f'Text inserted at line {command.insert_line} in {command.path}'  # pragma: no cover

        def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
            return f'File deleted: {command.path}'  # pragma: no cover

        def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
            return f'Renamed {command.old_path} to {command.new_path}'  # pragma: no cover

        def clear_all_memory(self) -> str:
            return 'All memory cleared'  # pragma: no cover

    fake_memory = FakeMemoryTool()

    @agent.tool_plain
    def memory(**command: Any) -> Any:
        return fake_memory.call(command)

    result = await agent.run('Where do I live?')
    assert result.output == snapshot("""\


According to my memory, you live in **Mexico City**.\
""")


async def test_anthropic_model_usage_limit_exceeded(
    allow_model_requests: None,
    anthropic_api_key: str,
):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 18 \\(input_tokens=19\\)',
    ):
        await agent.run(
            'The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=18, count_tokens_before_request=True),
        )


async def test_anthropic_model_usage_limit_not_exceeded(
    allow_model_requests: None,
    anthropic_api_key: str,
):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=25, count_tokens_before_request=True),
    )
    assert result.output == snapshot(
        """\
I noticed a small typo in that famous pangram! It should be:

"The quick brown fox jumps over the **lazy dog**."

(There should be a space between "lazy" and "dog")

This sentence is often used for testing typewriters, fonts, and keyboards because it contains every letter of the English alphabet at least once.\
"""
    )


async def test_anthropic_count_tokens_with_mock(allow_model_requests: None):
    """Test that count_tokens is called on the mock client."""
    c = completion_message(
        [BetaTextBlock(text='hello world', type='text')], BetaUsage(input_tokens=5, output_tokens=10)
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))
    assert result.output == 'hello world'
    assert len(mock_client.chat_completion_kwargs) == 2  # type: ignore
    count_tokens_kwargs = mock_client.chat_completion_kwargs[0]  # type: ignore
    assert 'model' in count_tokens_kwargs
    assert 'messages' in count_tokens_kwargs


async def test_anthropic_count_tokens_with_no_messages(allow_model_requests: None):
    """Test count_tokens when messages_ is None (no exception configured)."""
    mock_client = cast(AsyncAnthropic, MockAnthropic())
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    result = await m.count_tokens(
        [ModelRequest.user_text_prompt('hello')],
        None,
        ModelRequestParameters(),
    )

    assert result.input_tokens == 10


@pytest.mark.vcr()
async def test_anthropic_count_tokens_error(allow_model_requests: None, anthropic_api_key: str):
    """Test that errors convert to ModelHTTPError."""
    model_id = 'claude-does-not-exist'
    model = AnthropicModel(model_id, provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))

    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == model_id


async def test_anthropic_bedrock_count_tokens_not_supported(env: TestEnv):
    """Test that AsyncAnthropicBedrock raises UserError for count_tokens."""
    from anthropic import AsyncAnthropicBedrock

    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_region='us-east-1',
    )
    provider = AnthropicProvider(anthropic_client=bedrock_client)
    model = AnthropicModel('anthropic.claude-3-5-sonnet-20241022-v2:0', provider=provider)
    agent = Agent(model)

    with pytest.raises(UserError, match='AsyncAnthropicBedrock client does not support `count_tokens` api.'):
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))


@pytest.mark.vcr()
async def test_anthropic_cache_messages_real_api(allow_model_requests: None, anthropic_api_key: str):
    """Test that anthropic_cache_messages setting adds cache_control and produces cache usage metrics.

    This test uses a cassette to verify the cache behavior without making real API calls in CI.
    When run with real API credentials, it demonstrates that:
    1. The first call with a long context creates a cache (cache_write_tokens > 0)
    2. Follow-up messages in the same conversation can read from that cache (cache_read_tokens > 0)
    """
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        system_prompt='You are a helpful assistant.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,
        ),
    )

    # First call with a longer message - this will cache the message content
    result1 = await agent.run('Please explain what Python is and its main use cases. ' * 100)
    usage1 = result1.usage()

    # With anthropic_cache_messages, the first call should write cache for the last message
    # (cache_write_tokens > 0 indicates that caching occurred)
    assert usage1.requests == 1
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Continue the conversation - this message appends to history
    # The previous cached message should still be in the request
    result2 = await agent.run('Can you summarize that in one sentence?', message_history=result1.all_messages())
    usage2 = result2.usage()

    # The second call should potentially read from cache if the previous message is still cached
    # (cache_read_tokens > 0 when cache hit occurs)
    # (cache_write_tokens > 0 as new message is added to cache)
    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.cache_write_tokens > 0
    assert usage2.output_tokens > 0


async def test_anthropic_container_setting_explicit(allow_model_requests: None):
    """Test that anthropic_container setting passes explicit container config to API."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with explicit container config
    await agent.run('hello', model_settings=AnthropicModelSettings(anthropic_container={'id': 'container_abc123'}))

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == BetaContainerParams(id='container_abc123')


async def test_anthropic_container_from_message_history(allow_model_requests: None):
    """Test that container_id from message history is passed to subsequent requests."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock([c, c])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Create a message history with a container_id in provider_details
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]

    # Run with the message history
    await agent.run('follow up', message_history=history)

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == 'container_from_history'


async def test_anthropic_container_setting_false_ignores_history(allow_model_requests: None):
    """Test that anthropic_container=False ignores container_id from history."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Create a message history with a container_id
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_should_be_ignored'},
        ),
    ]

    # Run with anthropic_container=False to force fresh container
    await agent.run(
        'follow up', message_history=history, model_settings=AnthropicModelSettings(anthropic_container=False)
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    # When anthropic_container=False, container should be OMIT (filtered out before sending to API)
    from anthropic import omit as OMIT

    assert completion_kwargs.get('container') is OMIT


async def test_anthropic_container_id_from_stream_response(allow_model_requests: None):
    """Test that container_id is extracted from streamed response and stored in provider_details."""
    from datetime import datetime

    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=BetaUsage(input_tokens=5, output_tokens=0),
                container=BetaContainer(
                    id='container_from_stream',
                    expires_at=datetime(2025, 1, 1, 0, 0, 0),
                ),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='hello'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('hello') as result:
        response = await result.get_output()
        assert response == 'hello'

    # Check that container_id was captured in the response
    messages = result.all_messages()
    model_response = messages[-1]
    assert isinstance(model_response, ModelResponse)
    assert model_response.provider_details is not None
    assert model_response.provider_details.get('container_id') == 'container_from_stream'
    assert model_response.provider_details.get('finish_reason') == 'end_turn'


async def test_anthropic_system_prompts_and_instructions_ordering():
    """Test that instructions are appended after all system prompts in the system prompt string."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
        ),
    ]
    model_request_parameters = ModelRequestParameters(
        instruction_parts=[InstructionPart(content='Instructions content')],
    )

    system_prompt, anthropic_messages = await m._map_message(messages, model_request_parameters, {})  # pyright: ignore[reportPrivateUsage]

    # Verify system prompts and instructions are joined in order: system1, system2, instructions
    assert system_prompt == snapshot(
        [
            {
                'type': 'text',
                'text': """\
System prompt 1

System prompt 2\
""",
            },
            {'type': 'text', 'text': 'Instructions content'},
        ]
    )
    # Verify user message is in anthropic_messages
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]['role'] == 'user'


async def test_anthropic_malformed_tool_args_no_crash(allow_model_requests: None):
    """Test that malformed JSON tool args don't crash the Anthropic retry path.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4430.

    When a tool call has malformed JSON arguments, a RetryPromptPart is correctly
    created. But when the message history is re-sent to Anthropic, the previous
    tool call's args are parsed via args_as_dict() which raises ValueError on
    invalid JSON, crashing the retry flow before the model can self-correct.
    """
    bad_args = '{"query": "bad query", "file_ids":[4556]</parameter>\n<parameter name="limit": 8}'

    # First response: the model "fixes" the tool call and returns text
    fixed_response = completion_message(
        [BetaTextBlock(text='Here is the corrected result.', type='text')],
        BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(fixed_response)
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Construct message_history with a malformed tool call + retry prompt
    # exactly as described in the issue
    message_history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='search_knowledge',
                    tool_call_id='toolu_123',
                    args=bad_args,
                ),
            ],
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    tool_name='search_knowledge',
                    tool_call_id='toolu_123',
                    content='Invalid JSON: expected `,` or `}` at line 1 column 99',
                ),
            ],
        ),
    ]

    # This should NOT raise ValueError — args_as_dict() now gracefully handles
    # malformed JSON by returning {'INVALID_JSON': ...}, allowing the retry to proceed.
    result = await agent.run(
        'Please fix the tool call and try again.',
        message_history=message_history,
    )
    assert result.output == 'Here is the corrected result.'
    assert result.all_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_knowledge',
                        args="""\
{"query": "bad query", "file_ids":[4556]</parameter>
<parameter name="limit": 8}\
""",
                        tool_call_id='toolu_123',
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Invalid JSON: expected `,` or `}` at line 1 column 99',
                        tool_name='search_knowledge',
                        tool_call_id='toolu_123',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Please fix the tool call and try again.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Here is the corrected result.')],
                usage=RequestUsage(input_tokens=10, output_tokens=5, details={'input_tokens': 10, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Verify the INVALID_JSON wrapper was actually sent to the Anthropic API
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['messages'] == snapshot(
        [
            {
                'role': 'assistant',
                'content': [
                    {
                        'id': 'toolu_123',
                        'type': 'tool_use',
                        'name': 'search_knowledge',
                        'input': {
                            'INVALID_JSON': """\
{"query": "bad query", "file_ids":[4556]</parameter>
<parameter name="limit": 8}\
""",
                        },
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'toolu_123',
                        'type': 'tool_result',
                        'content': """\
Invalid JSON: expected `,` or `}` at line 1 column 99

Fix the errors and try again.\
""",
                        'is_error': True,
                    },
                    {'text': 'Please fix the tool call and try again.', 'type': 'text'},
                ],
            },
        ]
    )


async def test_pause_turn_streaming_continuation(allow_model_requests: None):
    """Test that pause_turn works with iter() + stream(), exercising ContinueRequestNode streaming."""
    from pydantic_ai._agent_graph import ContinueRequestNode
    from pydantic_graph import End

    # First response: pause_turn (non-streaming, via ModelRequestNode.run())
    c1 = completion_message([BetaTextBlock(text='first', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    c1.stop_reason = 'pause_turn'

    def _make_stream(text: str, stop_reason: str) -> list[BetaRawMessageStreamEvent]:
        return [
            BetaRawMessageStartEvent(
                type='message_start',
                message=BetaMessage(
                    id=f'msg_{text}',
                    model='claude-3-5-haiku-123',
                    role='assistant',
                    type='message',
                    content=[],
                    stop_reason=None,
                    usage=BetaUsage(input_tokens=10, output_tokens=0),
                ),
            ),
            BetaRawContentBlockStartEvent(
                type='content_block_start',
                index=0,
                content_block=BetaTextBlock(type='text', text=text),
            ),
            BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
            BetaRawMessageDeltaEvent(
                type='message_delta',
                delta=Delta(stop_reason=cast(Any, stop_reason)),
                usage=BetaMessageDeltaUsage(input_tokens=10, output_tokens=5),
            ),
            BetaRawMessageStopEvent(type='message_stop'),
        ]

    # stream[0] is a placeholder (index 0 uses messages_ path for the initial non-streaming request).
    # stream[1]: streaming continuation returns pause_turn again (covers line 1172).
    # stream[2]: streaming continuation returns end_turn (final).
    mock_client = cast(
        AsyncAnthropic,
        MockAnthropic(
            messages_=[c1],
            stream=[[], _make_stream('second', 'pause_turn'), _make_stream('done', 'end_turn')],
        ),
    )
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    continuation_count = 0
    async with agent.iter('test prompt') as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            if isinstance(node, ContinueRequestNode):
                continuation_count += 1
                # Exercise the streaming path on ContinueRequestNode
                async with node.stream(agent_run.ctx) as stream:
                    async for _event in stream:
                        pass
            # Advance: for ContinueRequestNode this returns cached _result (line 1118)
            node = await agent_run.next(node)

    assert continuation_count == 2
    assert agent_run.result
    assert agent_run.result.output == snapshot('firstseconddone')


async def test_pause_turn_streaming_continuation_stream_error(allow_model_requests: None):
    """Test that an error during streaming continuation propagates and cancels the wrap task."""
    from pydantic_ai._agent_graph import ContinueRequestNode
    from pydantic_graph import End

    c1 = completion_message([BetaTextBlock(text='first', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    c1.stop_reason = 'pause_turn'

    # Continuation stream that raises mid-stream
    error_stream: list[MockRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_err',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=10, output_tokens=0),
            ),
        ),
        RuntimeError('stream exploded'),
    ]

    mock_client = cast(
        AsyncAnthropic,
        MockAnthropic(messages_=[c1], stream=[[], error_stream]),
    )
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    async with agent.iter('test prompt') as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            if isinstance(node, ContinueRequestNode):
                with pytest.raises(RuntimeError, match='stream exploded'):
                    async with node.stream(agent_run.ctx) as stream:
                        async for _event in stream:
                            pass
                break
            node = await agent_run.next(node)


async def test_anthropic_local_shell_toolset(allow_model_requests: None, anthropic_api_key: str):
    """Test ShellToolset with Anthropic using native bash_20250124 format."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        toolsets=[ShellToolset.local()],
        instructions='Always use the shell tool. Keep responses brief.',
    )

    result = await agent.run('What is 2 + 2? Use python to calculate it.')
    assert '4' in result.output
    messages = result.all_messages()
    # Verify tool call used ToolCallPart (not BuiltinToolCallPart)
    tool_calls = [p for msg in messages for p in msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_calls) >= 1
    # Verify tool return exists
    tool_returns = [p for msg in messages for p in msg.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) >= 1
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2? Use python to calculate it.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='shell',
                        args={'command': 'python3 -c "print(2 + 2)"'},
                        tool_call_id='toolu_016EHVaf6tCPBWQSjkredb5c',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=774,
                    output_tokens=65,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 774,
                        'output_tokens': 65,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01XHMT9Qb389yy3bPgZuugz1',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='shell',
                        content='{"output": "4\\n", "exit_code": 0}',
                        tool_call_id='toolu_016EHVaf6tCPBWQSjkredb5c',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The result of **2 + 2 is 4**! 🎉')],
                usage=RequestUsage(
                    input_tokens=953,
                    output_tokens=22,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 953,
                        'output_tokens': 22,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01YDLfT9HyYBeaKP57V4kUwS',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_local_shell_toolset_stream(allow_model_requests: None, anthropic_api_key: str):
    """Test ShellToolset streaming with Anthropic using native bash_20250124 format."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        toolsets=[ShellToolset.local()],
        instructions='Always use the shell tool. Keep responses brief.',
    )

    async with agent.run_stream('What is 2 + 2? Use python to calculate it.') as result:
        output = await result.get_output()
    assert '4' in output
    messages = result.all_messages()
    tool_calls = [p for msg in messages for p in msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_calls) >= 1
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2? Use python to calculate it.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='shell',
                        args='{"command": "python3 -c \\"print(2 + 2)\\""}',
                        tool_call_id='toolu_015Rk6ndKaiXYx9AiJcpUR2H',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=774,
                    output_tokens=65,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 774,
                        'output_tokens': 65,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_016qmMxFjrLSYY2paN55KqFC',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='shell',
                        content='{"output": "4\\n", "exit_code": 0}',
                        tool_call_id='toolu_015Rk6ndKaiXYx9AiJcpUR2H',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Always use the shell tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The result of **2 + 2 = 4**! 🎉')],
                usage=RequestUsage(
                    input_tokens=953,
                    output_tokens=22,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 953,
                        'output_tokens': 22,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_011fHpYwuemfFdxg8x6fvsd8',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def _text_editor_execute(cmd: TextEditorCommand) -> TextEditorOutput:
    """Simple text editor executor for testing."""
    command = cmd['command']
    path = cmd['path']
    if command == 'view':
        return TextEditorOutput(output=f'Contents of {path}:\nprint("hello world")\n')
    elif command == 'create':
        return TextEditorOutput(output=f'File created: {path}')
    elif command == 'str_replace':
        return TextEditorOutput(output=f'Replacement performed in {path}')
    elif command == 'insert':
        return TextEditorOutput(output=f'Text inserted in {path}')
    else:
        return TextEditorOutput(output=f'Unknown command: {command}', success=False)


async def test_anthropic_text_editor_toolset(allow_model_requests: None, anthropic_api_key: str):
    """Test TextEditorToolset with Anthropic using native text_editor_20250728 format."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(
        m,
        toolsets=[TextEditorToolset(execute=_text_editor_execute, max_characters=100_000)],
        instructions='Always use the text editor tool. Keep responses brief.',
    )

    result = await agent.run('View the file /tmp/example.py')
    assert 'hello world' in result.output.lower() or 'example.py' in result.output.lower()
    messages = result.all_messages()
    tool_calls = [p for msg in messages for p in msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_calls) >= 1
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='View the file /tmp/example.py', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Always use the text editor tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='str_replace_based_edit_tool',
                        args={'command': 'view', 'path': '/tmp/example.py'},
                        tool_call_id='toolu_01F9jsn5Fg38kS5NSUURKpyy',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=1270,
                    output_tokens=82,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1270,
                        'output_tokens': 82,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_015xuYycvoFxnAxGM9ZDA7F8',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='str_replace_based_edit_tool',
                        content='{"output": "Contents of /tmp/example.py:\\nprint(\\"hello world\\")\\n", "success": true}',
                        tool_call_id='toolu_01F9jsn5Fg38kS5NSUURKpyy',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Always use the text editor tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The file `/tmp/example.py` contains a single line of Python code:

```python
print("hello world")
```

It's a simple script that prints `"hello world"` to the console.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=1391,
                    output_tokens=49,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1391,
                        'output_tokens': 49,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01GgDAYdcLVErjsGgMa56dgw',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    ('command', 'expected_output', 'expected_success'),
    [
        ('create', 'File created: /tmp/example.py', True),
        ('str_replace', 'Replacement performed in /tmp/example.py', True),
        ('insert', 'Text inserted in /tmp/example.py', True),
        ('unknown', 'Unknown command: unknown', False),
    ],
)
async def test_text_editor_execute_helper_variants(
    command: str,
    expected_output: str,
    expected_success: bool,
) -> None:
    result = await _text_editor_execute(cast(TextEditorCommand, {'command': command, 'path': '/tmp/example.py'}))
    assert result.output == expected_output
    assert result.success is expected_success


async def test_anthropic_text_editor_toolset_stream(allow_model_requests: None, anthropic_api_key: str):
    """Test TextEditorToolset streaming with Anthropic using native text_editor_20250728 format."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(
        m,
        toolsets=[TextEditorToolset(execute=_text_editor_execute)],
        instructions='Always use the text editor tool. Keep responses brief.',
    )

    async with agent.run_stream('View the file /tmp/example.py') as result:
        output = await result.get_output()
    assert 'hello world' in output.lower() or 'example.py' in output.lower()
    messages = result.all_messages()
    tool_calls = [p for msg in messages for p in msg.parts if isinstance(p, ToolCallPart)]
    assert len(tool_calls) >= 1
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='View the file /tmp/example.py', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='Always use the text editor tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='str_replace_based_edit_tool',
                        args='{"command": "view", "path": "/tmp/example.py"}',
                        tool_call_id='toolu_015czvgRAN7XL7DZXTBbSXiP',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=1270,
                    output_tokens=82,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1270,
                        'output_tokens': 82,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_0163z5LYP99BxFSy96FRYrVQ',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='str_replace_based_edit_tool',
                        content='{"output": "Contents of /tmp/example.py:\\nprint(\\"hello world\\")\\n", "success": true}',
                        tool_call_id='toolu_015czvgRAN7XL7DZXTBbSXiP',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Always use the text editor tool. Keep responses brief.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The file `/tmp/example.py` contains a single line of Python code:

```python
print("hello world")
```

It's a simple script that prints **"hello world"** to the console.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=1391,
                    output_tokens=49,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1391,
                        'output_tokens': 49,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01W31jAUhgU7aDqWUQK1VHgy',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_shell_tool_skills_are_sent_in_container_and_betas(allow_model_requests: None):
    from pydantic_ai.builtin_tools import SkillReference

    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[ShellTool(skills=[SkillReference(skill_id='computer-use', version='2', source='provider')])],
    )

    await agent.run('hello')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container']['skills'] == [
        {'skill_id': 'computer-use', 'type': 'anthropic', 'version': '2'}
    ]
    assert 'skills-2025-10-02' in completion_kwargs['betas']


async def test_anthropic_text_editor_toolset_sends_max_characters(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent: Agent[None, str] = Agent(
        m, toolsets=[TextEditorToolset(execute=_text_editor_execute, max_characters=100_000)]
    )

    await agent.run('View /tmp/example.py')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert {
        'type': 'text_editor_20250728',
        'name': 'str_replace_based_edit_tool',
        'max_characters': 100_000,
    } in completion_kwargs['tools']


def _shell_tool_history_round_trip_cases() -> list[Any]:
    if not imports_successful():
        return []

    cases = [
        pytest.param(
            'text_editor_code_execution',
            {'command': 'view', 'path': '/tmp/example.py'},
            BetaTextEditorCodeExecutionToolResultBlock(
                tool_use_id='srv_123',
                type='text_editor_code_execution_tool_result',
                content={  # pyright: ignore[reportArgumentType]
                    'type': 'text_editor_code_execution_view_result',
                    'content': 'print("hello world")',
                    'file_type': 'text',
                },
            ),
            'text_editor_code_execution_tool_result',
            id='text-editor',
        )
    ]

    try:
        from anthropic.types.beta import BetaBashCodeExecutionToolResultBlock
    except ImportError:  # pragma: no cover — guard for older SDK versions
        return cases

    return [
        pytest.param(
            'bash_code_execution',
            {'command': 'python3 -c "print(4)"'},
            BetaBashCodeExecutionToolResultBlock(
                tool_use_id='srv_123',
                type='bash_code_execution_tool_result',
                content={  # pyright: ignore[reportArgumentType]
                    'content': [],
                    'return_code': 0,
                    'stderr': '',
                    'stdout': '4\n',
                    'type': 'bash_code_execution_result',
                },
            ),
            'bash_code_execution_tool_result',
            id='bash',
        ),
        *cases,
    ]


@pytest.mark.parametrize(
    ('server_tool_name', 'server_tool_args', 'result_block', 'result_type'),
    _shell_tool_history_round_trip_cases(),
)
async def test_anthropic_shell_tool_history_round_trip_preserves_server_tool_name(
    server_tool_name: str,
    server_tool_args: dict[str, Any],
    result_block: BetaContentBlock,
    result_type: str,
    allow_model_requests: None,
):
    first_response = completion_message(
        [
            BetaServerToolUseBlock(
                id='srv_123',
                input=server_tool_args,
                name=server_tool_name,  # pyright: ignore[reportArgumentType]
                type='server_tool_use',
            ),
            result_block,
            BetaTextBlock(text='done', type='text'),
        ],
        BetaUsage(input_tokens=5, output_tokens=5),
    )
    second_response = completion_message(
        [BetaTextBlock(text='summary', type='text')], BetaUsage(input_tokens=5, output_tokens=5)
    )
    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[ShellTool()])

    first_result = await agent.run('Run the shell tool')
    await agent.run('Summarize the previous tool result', message_history=first_result.all_messages())

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[1]
    assistant_message: dict[str, Any] = completion_kwargs['messages'][1]
    content: list[dict[str, Any]] = assistant_message['content']
    assert assistant_message['role'] == 'assistant'
    assert any(
        item.get('type') == 'server_tool_use'
        and item.get('name') == server_tool_name
        and item.get('input') == server_tool_args
        for item in content
    )
    assert any(item.get('type') == result_type and item.get('tool_use_id') == 'srv_123' for item in content)


@pytest.mark.parametrize(
    ('target', 'builtin_tools', 'expected_block_type'),
    [
        pytest.param('container', [], None, id='no-shell-tool'),
        pytest.param('container', [ShellTool()], 'container_upload', id='shell-tool'),
    ],
)
async def test_anthropic_uploaded_file_container_target_requires_shell_tool(
    target: str,
    builtin_tools: list[ShellTool],
    expected_block_type: str | None,
    allow_model_requests: None,
):
    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=builtin_tools)
    prompt = ['Use this file', UploadedFile(file_id='file-container-123', provider_name='anthropic', target=target)]  # pyright: ignore[reportArgumentType]

    if expected_block_type is None:
        with pytest.raises(UserError, match="`UploadedFile` with `target='container'` requires a `ShellTool`"):
            await agent.run(prompt)
    else:
        await agent.run(prompt)
        completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
        assert completion_kwargs['messages'][0]['content'] == [
            {'text': 'Use this file', 'type': 'text'},
            {'file_id': 'file-container-123', 'type': expected_block_type},
        ]


async def test_anthropic_streamed_text_editor_tool_result_mapping(allow_model_requests: None):
    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=BetaUsage(input_tokens=5, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextEditorCodeExecutionToolResultBlock(
                tool_use_id='srv_123',
                type='text_editor_code_execution_tool_result',
                content={  # pyright: ignore[reportArgumentType]
                    'type': 'text_editor_code_execution_view_result',
                    'content': 'print("hello world")',
                    'file_type': 'text',
                },
            ),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]
    mock_client = MockAnthropic.create_stream_mock(stream_events)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    async with m.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='hello')])],
        None,
        ModelRequestParameters(),
    ) as response:
        async for _ in response:
            pass
        mapped = response.get()

    assert len(mapped.parts) == 1
    part = mapped.parts[0]
    assert isinstance(part, BuiltinToolReturnPart)
    assert part.tool_name == 'shell'
    assert part.tool_call_id == 'srv_123'
    assert part.provider_name == 'anthropic'
    assert part.content == {
        'type': 'text_editor_code_execution_view_result',
        'content': 'print("hello world")',
        'file_type': 'text',
        'num_lines': None,
        'start_line': None,
        'total_lines': None,
    }


@pytest.mark.parametrize(
    ('command', 'path', 'expected_output', 'expected_success'),
    [
        ('create', '/tmp/create.py', 'File created: /tmp/create.py', True),
        ('str_replace', '/tmp/edit.py', 'Replacement performed in /tmp/edit.py', True),
        ('insert', '/tmp/insert.py', 'Text inserted in /tmp/insert.py', True),
        ('unknown', '/tmp/unknown.py', 'Unknown command: unknown', False),
    ],
)
async def test_text_editor_execute_helper_covers_all_command_variants(
    command: str,
    path: str,
    expected_output: str,
    expected_success: bool,
):
    output = await _text_editor_execute(cast(TextEditorCommand, {'command': command, 'path': path}))

    assert output.output == expected_output
    assert output.success is expected_success


@pytest.mark.parametrize(
    ('toolset', 'warning_kind'),
    [
        pytest.param(ShellToolset.local(), 'shell', id='shell'),
        pytest.param(
            TextEditorToolset(execute=_text_editor_execute, max_characters=100_000),
            'text_editor',
            id='text_editor',
        ),
    ],
)
async def test_anthropic_warns_when_native_tool_falls_back(
    toolset: Any,
    warning_kind: str,
    allow_model_requests: None,
) -> None:
    import pydantic_ai.models as models_mod
    from pydantic_ai.profiles import ModelProfile

    models_mod.NATIVE_TOOL_FALLBACK_WARNED.clear()

    profile = ModelProfile(
        supports_native_shell_tool=False,
        supports_native_text_editor_tool=False,
    )
    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client), profile=profile)
    agent = Agent(m, toolsets=[toolset])

    with pytest.warns(UserWarning, match=f'Native `{warning_kind}` tool: falling back to function tool format'):
        await agent.run('hello')

    # Verify fallback to function tool format in outgoing request
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tool_types = {t.get('type', 'custom') for t in completion_kwargs['tools']}
    assert 'custom' in tool_types  # function tool, not native

    # Second call should NOT warn again
    c2 = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client2 = MockAnthropic.create_mock(c2)
    m2 = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client2), profile=profile)
    agent2 = Agent(m2, toolsets=[toolset])

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        await agent2.run('hello')

    assert not any(warning_kind in str(w.message) for w in caught)


async def test_anthropic_shell_tool_skills_version_none_omits_version_key(allow_model_requests: None):
    """When a SkillReference has version=None, the mapped BetaSkillParams should not include a 'version' key."""
    from pydantic_ai.builtin_tools import SkillReference

    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[ShellTool(skills=[SkillReference(skill_id='data-analysis', source='provider')])],
    )

    await agent.run('hello')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    skills = completion_kwargs['container']['skills']
    assert skills == snapshot([{'skill_id': 'data-analysis', 'type': 'anthropic'}])


async def test_anthropic_shell_tool_skills_added_to_explicit_container(allow_model_requests: None):
    """When an explicit anthropic_container is provided alongside ShellTool skills, skills are merged into it."""
    from pydantic_ai.builtin_tools import SkillReference

    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        builtin_tools=[ShellTool(skills=[SkillReference(skill_id='computer-use', version='2', source='provider')])],
    )

    await agent.run(
        'hello',
        model_settings=AnthropicModelSettings(anthropic_container=BetaContainerParams(id='container_existing')),
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    container = completion_kwargs['container']
    assert container == snapshot(
        {'id': 'container_existing', 'skills': [{'skill_id': 'computer-use', 'type': 'anthropic', 'version': '2'}]}
    )


async def test_anthropic_container_setting_string(allow_model_requests: None):
    """Test that anthropic_container as a plain string passes the string directly to API."""
    c = completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    await agent.run('hello', model_settings=cast(AnthropicModelSettings, {'anthropic_container': 'cntr_string_123'}))

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == 'cntr_string_123'


async def test_anthropic_container_setting_explicit_stream(allow_model_requests: None):
    """Streaming with explicit anthropic_container skips the history-container check."""
    usage = BetaUsage(input_tokens=5, output_tokens=5)
    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_stream',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=usage,
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(type='text', text='ok'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]
    mock_client = MockAnthropic.create_stream_mock(stream_events)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream(
        'hello', model_settings=AnthropicModelSettings(anthropic_container={'id': 'container_explicit'})
    ) as result:
        output = await result.get_output()
    assert output == 'ok'

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == BetaContainerParams(id='container_explicit')


async def test_anthropic_container_error_retry_bash(allow_model_requests: None):
    """Auto-retry with a fresh container when all bash exec results are errors (non-streaming)."""
    from anthropic.types.beta import BetaBashCodeExecutionToolResultBlock, BetaBashCodeExecutionToolResultError

    usage = BetaUsage(input_tokens=5, output_tokens=5)
    error_block = BetaBashCodeExecutionToolResultBlock(
        type='bash_code_execution_tool_result',
        content=BetaBashCodeExecutionToolResultError(
            type='bash_code_execution_tool_result_error',
            error_code='invalid_tool_input',
        ),
        tool_use_id='tu_001',
    )
    error_response = completion_message([error_block], usage)
    ok_response = completion_message([BetaTextBlock(text='retried ok', type='text')], usage)
    mock_client = MockAnthropic.create_mock([error_response, ok_response])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'retried ok'

    # Two requests made: original + retry
    kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(kwargs) == 2


async def test_anthropic_container_error_retry_text_editor(allow_model_requests: None):
    """Auto-retry when all text editor exec results are errors (covers _is_exec_result_error for text editor)."""
    from anthropic.types.beta import BetaTextEditorCodeExecutionToolResultError

    usage = BetaUsage(input_tokens=5, output_tokens=5)
    error_block = BetaTextEditorCodeExecutionToolResultBlock(
        type='text_editor_code_execution_tool_result',
        content=BetaTextEditorCodeExecutionToolResultError(
            type='text_editor_code_execution_tool_result_error',
            error_code='invalid_tool_input',
        ),
        tool_use_id='tu_002',
    )
    error_response = completion_message([error_block], usage)
    ok_response = completion_message([BetaTextBlock(text='retried ok', type='text')], usage)
    mock_client = MockAnthropic.create_mock([error_response, ok_response])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'retried ok'

    kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(kwargs) == 2


async def test_anthropic_stream_container_error_retry(allow_model_requests: None):
    """Streaming auto-retry when history container produces all error exec results."""
    from anthropic.types.beta import BetaBashCodeExecutionToolResultBlock, BetaBashCodeExecutionToolResultError

    usage = BetaUsage(input_tokens=5, output_tokens=5)

    # Stream events with a bash execution error block
    error_block = BetaBashCodeExecutionToolResultBlock(
        type='bash_code_execution_tool_result',
        content=BetaBashCodeExecutionToolResultError(
            type='bash_code_execution_tool_result_error',
            error_code='invalid_tool_input',
        ),
        tool_use_id='tu_err',
    )
    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_err',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=usage,
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=error_block,
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    # Non-streaming retry response
    ok_response = completion_message([BetaTextBlock(text='retried ok', type='text')], usage)

    # Construct mock with both stream (for first call) and messages (for retry)
    # Index 0: stream call reads stream[0], index becomes 1
    # Index 1: non-stream retry reads messages_[1], index becomes 2
    dummy_response = completion_message([BetaTextBlock(text='dummy', type='text')], usage)
    mock = MockAnthropic(
        stream=[stream_events],
        messages_=[dummy_response, ok_response],
    )
    mock_client = cast(AsyncAnthropic, mock)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Message history with a container_id so has_history_container is True
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'cntr_from_history'},
        ),
    ]

    async with agent.run_stream('follow up', message_history=history) as result:
        output = await result.get_output()
        assert output == 'retried ok'

    # Two requests: stream + non-stream retry
    kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(kwargs) == 2


async def test_anthropic_stream_history_container_success(allow_model_requests: None):
    """Streaming with history container succeeds when no container errors in response."""
    usage = BetaUsage(input_tokens=5, output_tokens=5)

    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_ok',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=usage,
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='streamed ok'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Message history with a container_id so has_history_container is True
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'cntr_from_history'},
        ),
    ]

    async with agent.run_stream('follow up', message_history=history) as result:
        output = await result.get_output()
        assert output == 'streamed ok'

    # Only one request (no retry)
    kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(kwargs) == 1


async def test_anthropic_code_execution_file_outputs(allow_model_requests: None, anthropic_api_key: str):
    """Code execution that produces file outputs extracts UploadedFile parts."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        instructions='Always use code execution. Create a small CSV file and save it. Keep responses brief.',
    )

    result = await agent.run('Create a CSV file with 3 rows of sample data (name, age) and save it as output.csv')
    messages = result.all_messages()

    # Check that UploadedFile parts were extracted from file outputs
    uploaded_files = [part for msg in messages for part in msg.parts if isinstance(part, UploadedFile)]
    # Code execution may or may not produce file outputs depending on the model's response
    # but the BuiltinToolReturnPart should be present
    tool_returns = [part for msg in messages for part in msg.parts if isinstance(part, BuiltinToolReturnPart)]
    assert len(tool_returns) >= 1

    # If file outputs were produced, verify they are UploadedFile with correct provider
    for uploaded_file in uploaded_files:
        assert uploaded_file.provider_name == 'anthropic'
        assert uploaded_file.file_id  # non-empty


async def test_anthropic_code_execution_file_outputs_stream(allow_model_requests: None, anthropic_api_key: str):
    """Streaming code execution that produces file outputs extracts UploadedFile parts."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        instructions='Always use code execution. Create a small CSV file and save it. Keep responses brief.',
    )

    async with agent.run_stream(
        'Create a CSV file with 3 rows of sample data (name, age) and save it as output.csv'
    ) as result:
        await result.get_output()

    messages = result.all_messages()

    # Check that UploadedFile parts were extracted from file outputs
    uploaded_files = [part for msg in messages for part in msg.parts if isinstance(part, UploadedFile)]
    tool_returns = [part for msg in messages for part in msg.parts if isinstance(part, BuiltinToolReturnPart)]
    assert len(tool_returns) >= 1

    for uploaded_file in uploaded_files:
        assert uploaded_file.provider_name == 'anthropic'
        assert uploaded_file.file_id


async def test_anthropic_shell_tool_file_outputs_stream(allow_model_requests: None, anthropic_api_key: str):
    """Streaming shell tool (bash code execution) that produces file outputs extracts UploadedFile parts."""
    m = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        builtin_tools=[ShellTool()],
        instructions='Always use the shell tool. Create a file and save it. Keep responses brief.',
    )

    async with agent.run_stream(
        'Use python to create a CSV file with 3 rows of data (name, age) and save it as output.csv'
    ) as result:
        await result.get_output()

    messages = result.all_messages()
    tool_returns = [part for msg in messages for part in msg.parts if isinstance(part, BuiltinToolReturnPart)]
    assert len(tool_returns) >= 1
    # Shell tool file outputs may or may not be present depending on model behavior
    uploaded_files = [part for msg in messages for part in msg.parts if isinstance(part, UploadedFile)]
    for uploaded_file in uploaded_files:
        assert uploaded_file.provider_name == 'anthropic'
        assert uploaded_file.file_id
