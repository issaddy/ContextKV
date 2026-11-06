"""LlamaIndex dialogue/compact reference with project-owned execution.

Only upstream history memory, question condensation and CompactAndRefine are
used here. Callbacks own retrieval, Qwen requests, actual costs and the one QA
commit. A call's source_blocks/reference_map form its retrieval catalogue;
messages are its exact visible input, including the current compact/refine slice.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Awaitable, Callable

from controller.evidence import assign_references
from controller.schema import EvidenceBlock, EvidenceKey, Message, TurnRecord


LLAMAINDEX_SOURCE_COMMIT = "f12d46acab73f5b2243ef49c2f00101617b38ce4"


@dataclass(frozen=True)
class LlamaIndexCall:
    step_id: str
    stage: str
    messages: tuple[Message, ...]
    output_budget: int
    source_blocks: tuple[EvidenceBlock, ...]
    reference_map: tuple[tuple[str, EvidenceKey], ...]


@dataclass(frozen=True)
class LlamaIndexDialogueResult:
    answer: str
    condensed_query: str
    source_blocks: tuple[EvidenceBlock, ...]
    reference_map: tuple[tuple[str, EvidenceKey], ...]
    calls: tuple[LlamaIndexCall, ...]
    framework_version: str


async def run_dialogue(
    question: str,
    history: tuple[TurnRecord, ...],
    *,
    retrieve_callback: Callable[[str], Awaitable[tuple[EvidenceBlock, ...]]],
    complete_callback: Callable[[LlamaIndexCall], Awaitable[str]],
    tokenizer,
    context_window: int,
    output_tokens: int,
    history_tokens: int,
    system_prompt: str,
    model_name: str = "Qwen3-32B",
) -> LlamaIndexDialogueResult:
    """Execute one external QA; every model call uses the supplied executor."""
    framework_version = version("llama-index-core")
    from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
    from llama_index.core.indices.prompt_helper import PromptHelper
    from llama_index.core.llms import CustomLLM
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
    from llama_index.core.memory import Memory
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode
    from sqlalchemy.ext.asyncio import create_async_engine

    source_blocks, reference_map, condensed_query = (), (), ""
    calls = []

    async def dispatch(stage, messages):
        call = LlamaIndexCall(f"llamaindex:{stage}:{len(calls)}", stage,
            tuple(Message(message.role.value, message.content) for message in messages),
            output_tokens, source_blocks, reference_map)
        calls.append(call)
        return await complete_callback(call)

    class CallbackLLM(CustomLLM):
        stage: str

        @property
        def metadata(self):
            return LLMMetadata(context_window=context_window, num_output=output_tokens,
                               is_chat_model=True, model_name=model_name)

        def complete(self, prompt, formatted=False, **kwargs):
            raise NotImplementedError("the reference uses asynchronous project execution")

        def stream_complete(self, prompt, formatted=False, **kwargs):
            raise NotImplementedError("the reference uses nonstreaming project execution")

        @llm_chat_callback()
        async def achat(self, messages, **kwargs):
            text = await dispatch(self.stage, messages)
            return ChatResponse(message=ChatMessage(role="assistant", content=text))

        @llm_completion_callback()
        async def acomplete(self, prompt, formatted=False, **kwargs):
            text = await dispatch(self.stage, (ChatMessage(role="user", content=prompt),))
            return CompletionResponse(text=text)

    class CallbackRetriever(BaseRetriever):
        def _retrieve(self, query_bundle):
            raise NotImplementedError("the reference uses asynchronous project retrieval")

        async def _aretrieve(self, query_bundle):
            nonlocal source_blocks, reference_map, condensed_query
            condensed_query = query_bundle.query_str
            source_blocks = tuple(await retrieve_callback(condensed_query))
            reference_map = assign_references(block.key for block in source_blocks)
            references = {key: reference for reference, key in reference_map}
            return [NodeWithScore(node=TextNode(id_=references[block.key], text=block.text,
                metadata={"citation_id": references[block.key], "title": block.title,
                          "version": block.key.version})) for block in source_blocks]

    callback_manager = CallbackManager([])
    condense_llm = CallbackLLM(stage="condense", callback_manager=callback_manager)
    answer_llm = CallbackLLM(stage="compact", system_prompt=system_prompt, callback_manager=callback_manager)
    synthesizer = get_response_synthesizer(llm=answer_llm,
        prompt_helper=PromptHelper(context_window=context_window, num_output=output_tokens,
                                   tokenizer=tokenizer.encode),
        response_mode=ResponseMode.COMPACT, callback_manager=callback_manager, use_async=True)
    query_engine = RetrieverQueryEngine(retriever=CallbackRetriever(callback_manager=callback_manager),
                                        response_synthesizer=synthesizer, callback_manager=callback_manager)
    database = create_async_engine("sqlite+aiosqlite:///:memory:")
    memory = Memory.from_defaults(token_limit=history_tokens, tokenizer_fn=tokenizer.encode,
        chat_history_token_ratio=1.0, memory_blocks=[], async_engine=database)
    try:
        await memory.aset([ChatMessage(role=role, content=text) for turn in history
                           for role, text in (("user", turn.user_text), ("assistant", turn.assistant_text))])
        chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine,
            llm=condense_llm, memory=memory)
        response = await chat_engine.achat(question)
    finally:
        await database.dispose()
    return LlamaIndexDialogueResult(str(response), condensed_query, source_blocks, reference_map,
                                   tuple(calls), framework_version)
