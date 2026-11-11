"""History-aware retrieval, ranked context concatenation, one answer call."""
from __future__ import annotations

from .dialogue import DialogueCall, DialogueResult
from controller.evidence import assign_references
from controller.schema import EvidenceSpan, Message, history_messages


async def run_dialogue(question, history, *, retrieve_callback, complete_callback,
                       template, context_window, output_tokens, history_tokens, system_prompt):
    limit = context_window-output_tokens
    messages, groups, used = (), [], 0
    for turn in reversed(history):
        group = history_messages((turn,))
        cost = len(template.render(group).input_ids)
        if used+cost > history_tokens:
            break
        groups.insert(0, group)
        messages = (*group, *messages)
        used += cost
    system = Message("system", system_prompt)
    while messages and len(template.render((system, *messages, Message("user", question))).input_ids) > limit:
        messages = messages[len(groups.pop(0)):]
    calls = []
    query = question
    if messages:
        rewrite = Message("system", "Rewrite the latest question as a standalone retrieval query using the "
            "conversation. Preserve its conditions and version. Return only the rewritten question.")
        rewrite_messages = (rewrite, *messages, Message("user", question))
        while len(template.render(rewrite_messages).input_ids) > limit and messages:
            messages = messages[len(groups.pop(0)):]
            rewrite_messages = (rewrite, *messages, Message("user", question))
        call = DialogueCall("rag:condense", "condense", rewrite_messages, output_tokens, (), ())
        calls.append(call)
        query = await complete_callback(call)
    retrieved = tuple(await retrieve_callback(query))
    blocks = tuple({block.key: block for block in retrieved}.values())
    references = dict((key, ref) for ref, key in assign_references(block.key for block in blocks))

    def render(pieces):
        content, spans = "", []
        for block, size in pieces:
            ref = references[block.key]
            content += f"\n[{ref}] {block.title} (version {block.key.version})\n"
            start = len(content)
            content += block.text[:size]
            base = block.span[0] if block.span else 0
            spans.append(EvidenceSpan(block.key, len(messages)+1, start, len(content),
                block.source_uri, block.provenance, ref, source_start=base, source_end=base+size))
            content += "\n"
        allowed = ", ".join(f"[{references[block.key]}]" for block, _ in pieces) or "none"
        content += f"\nCitations allowed for this answer: {allowed}.\nQuestion:\n{question}\n"
        prompt = (system, *messages, Message("user", content))
        return prompt, tuple(spans)

    while messages and len(template.render(render(())[0]).input_ids) > limit:
        messages = messages[len(groups.pop(0)):]
    pieces = []
    for block in blocks:
        trial = (*pieces, (block, len(block.text)))
        if len(template.render(render(trial)[0]).input_ids) <= limit:
            pieces.append((block, len(block.text)))
            continue
        low, high = 0, len(block.text)
        while low < high:
            middle = (low+high+1)//2
            if len(template.render(render((*pieces, (block, middle)))[0]).input_ids) <= limit:
                low = middle
            else:
                high = middle-1
        if low:
            pieces.append((block, low))
        break
    final_messages, spans = render(pieces)
    shown = tuple(block for block, _ in pieces)
    reference_map = tuple((references[block.key], block.key) for block in shown)
    call = DialogueCall("rag:answer", "answer", final_messages, output_tokens, shown, reference_map, spans)
    calls.append(call)
    answer = await complete_callback(call)
    return DialogueResult(answer, query, shown, reference_map, tuple(calls), "ordinary-rag")
