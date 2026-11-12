"""Temporary official benchmark scoring input."""
from controller.schema import OnlineTurn


def official_record(turn: OnlineTurn, answer: str) -> dict:
    """Export the current reading material and preserve raw retrieval separately.

    Preserve the supplied source set. Published MTRAG-UN reference tasks can have
    more contexts than the competition submission checker permits; that check
    belongs to submission validation, not this lossless result export.
    """
    contexts = [{"document_id": c.document_id, "source": c.source, "score": c.score or 0.0,
                 "text": c.text, "title": c.title} for c in (*turn.contexts, *turn.supporting_contexts)]
    messages = [{"speaker": "user" if m.speaker == "user" else "agent", "text": m.text} for m in turn.history]
    messages.append({"speaker": "user", "text": turn.query})
    record = {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
            "Collection": turn.collection, "input": messages,
            "contexts": contexts, "predictions": [{"text": answer}]}
    if turn.retrieval_contexts is not None:
        record["retrieval_contexts"] = [{"document_id": c.document_id, "source": c.source,
            "score": c.score or 0.0, "text": c.text, "title": c.title} for c in turn.retrieval_contexts]
    elif turn.supporting_contexts:
        record["retrieval_contexts"] = contexts[:len(turn.contexts)]
    return record
