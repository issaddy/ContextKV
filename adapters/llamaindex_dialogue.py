"""LlamaIndex dialogue/compact reference with project-owned execution.

Only upstream history memory, question condensation and CompactAndRefine are
used here. Callbacks own retrieval, Qwen requests, actual costs and the one QA
commit. A call's source_blocks/reference_map form its retrieval catalogue;
messages are its exact visible input, including the current compact/refine slice.
"""
from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version
from typing import Awaitable, Callable

from controller.evidence import assign_references
from controller.schema import EvidenceBlock, EvidenceSpan, Message, TurnRecord, history_messages
from .dialogue import DialogueCall, DialogueResult


LLAMAINDEX_SOURCE_COMMIT = "f12d46acab73f5b2243ef49c2f00101617b38ce4"


async def run_dialogue(
    question: str,
    history: tuple[TurnRecord, ...],
    *,
    retrieve_callback: Callable[[str], Awaitable[tuple[EvidenceBlock, ...]]],
    complete_callback: Callable[[DialogueCall], Awaitable[str]],
    tokenizer,
    context_window: int,
    output_tokens: int,
    history_tokens: int,
    system_prompt: str,
    model_name: str = "Qwen3-32B",
) -> DialogueResult:
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
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode
    from sqlalchemy.ext.asyncio import create_async_engine

    source_blocks, reference_map, condensed_query = (), (), ""
    calls = []

    # Observe upstream packing without replacing its history or compact algorithm.
    packed_sources = {}
    context_open, context_close = "<llamaindex_context>\n", "\n</llamaindex_context>"

    class AuditedPromptHelper(PromptHelper):
        def repack(self, prompt, text_chunks, **kwargs):
            combined, spans = "", []
            for text in text_chunks:
                stripped = text.strip()
                if not stripped:
                    continue
                offset = len(combined)+(2 if combined else 0)
                left = len(text)-len(text.lstrip())
                combined += ("\n\n" if combined else "")+stripped
                for span in packed_sources[text]:
                    first, last = max(span.start, left), min(span.end, left+len(stripped))
                    if first < last:
                        spans.append(replace(span, start=offset+first-left, end=offset+last-left,
                            source_start=span.source_start+first-span.start,
                            source_end=span.source_start+last-span.start))
            result = super().repack(prompt, text_chunks, **kwargs)
            previous_start = -1
            for text in result:
                start = combined.find(text, previous_start+1)
                if start < 0:
                    raise ValueError("upstream packed text lost its original source coordinates")
                previous_start = start
                clipped = []
                for span in spans:
                    first, last = max(span.start, start), min(span.end, start+len(text))
                    if first < last:
                        clipped.append(replace(span, start=first-start, end=last-start,
                            source_start=span.source_start+first-span.start,
                            source_end=span.source_start+last-span.start))
                packed_sources[text] = tuple(clipped)
            return result

    async def dispatch(stage, messages):
        messages = tuple(Message(message.role.value, message.content or "") for message in messages)
        spans = []
        if stage == "compact":
            for index, message in enumerate(messages):
                if context_open not in message.content:
                    continue
                first = message.content.index(context_open)+len(context_open)
                last = message.content.index(context_close, first)
                context = message.content[first:last]
                spans.extend(replace(span, message_index=index, start=first+span.start,
                    end=first+span.end) for span in packed_sources[context])
                stage = "refine" if "<llamaindex_existing_answer>" in message.content else "compact"
        call = DialogueCall(f"llamaindex:{stage}:{len(calls)}", stage, messages,
            output_tokens, source_blocks, reference_map, tuple(spans))
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
            nodes = []
            for block in source_blocks:
                reference = references[block.key]
                node = TextNode(id_=reference, text=block.text, metadata={
                    "citation_id": reference, "title": block.title, "version": block.key.version})
                content = node.get_content(metadata_mode=MetadataMode.LLM)
                # Upstream strips the formatted metadata + body. The nonempty
                # citation metadata preserves leading body whitespace, while
                # trailing body whitespace is absent from the actual LLM input.
                body = node.get_content(metadata_mode=MetadataMode.NONE).rstrip()
                start = len(content)-len(body)
                if content[start:] != body:
                    raise ValueError("upstream source rendering lost its original body")
                packed_sources[content] = (EvidenceSpan(block.key, 0, start, len(content),
                    block.source_uri, block.provenance, reference,
                    source_start=block.span[0] if block.span else 0,
                    source_end=(block.span[0] if block.span else 0)+len(body)),) if body else ()
                nodes.append(NodeWithScore(node=node))
            return nodes

    callback_manager = CallbackManager([])
    condense_llm = CallbackLLM(stage="condense", callback_manager=callback_manager)
    answer_llm = CallbackLLM(stage="compact", system_prompt=system_prompt, callback_manager=callback_manager)
    synthesizer = get_response_synthesizer(llm=answer_llm,
        prompt_helper=AuditedPromptHelper(context_window=context_window, num_output=output_tokens,
                                   tokenizer=tokenizer.encode),
        text_qa_template=PromptTemplate(
            "Context information is below.\n"+context_open+"{context_str}"+context_close+
            "\nGiven the context information, answer the query.\nQuery: {query_str}\nAnswer: "),
        refine_template=PromptTemplate(
            "Query: {query_str}\n<llamaindex_existing_answer>\n{existing_answer}\n"
            "</llamaindex_existing_answer>\nRefine the existing answer using this additional context.\n"+
            context_open+"{context_msg}"+context_close+
            "\nIf the context is not useful, return the existing answer.\nRefined answer: "),
        response_mode=ResponseMode.COMPACT, callback_manager=callback_manager, use_async=True)
    query_engine = RetrieverQueryEngine(retriever=CallbackRetriever(callback_manager=callback_manager),
                                        response_synthesizer=synthesizer, callback_manager=callback_manager)
    database = create_async_engine("sqlite+aiosqlite:///:memory:")
    memory = Memory.from_defaults(token_limit=history_tokens, tokenizer_fn=tokenizer.encode,
        chat_history_token_ratio=1.0, memory_blocks=[], async_engine=database)
    try:
        await memory.aset([ChatMessage(role=message.role, content=message.content)
                           for message in history_messages(history)])
        chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=query_engine,
            llm=condense_llm, memory=memory)
        response = await chat_engine.achat(question)
    finally:
        await database.dispose()
    return DialogueResult(str(response), condensed_query, source_blocks, reference_map,
                                   tuple(calls), framework_version)
