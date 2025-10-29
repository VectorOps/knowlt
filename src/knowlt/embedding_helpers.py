from typing import TYPE_CHECKING
import asyncio
from pydantic import BaseModel

from knowlt.data import NodeFilter
from knowlt.logger import logger
from knowlt.models import ModelId, Vector, Repo
from knowlt.project import ProjectManager
# Save updates in batches as results complete
EMBEDDING_UPDATE_BATCH_SIZE = 200


class SymbolEmbeddingItem(BaseModel):
    symbol_id: ModelId
    body: str
# Embedding helpers
async def schedule_symbol_embedding(
    node_repo, emb_calc, items: list[SymbolEmbeddingItem]
) -> None:
    """
    Enqueue embeddings for all items concurrently. As each result completes,
    buffer updates and flush them to the data store in batches.
    Returns only after all items have been processed.
    """
    if not items:
        return

    model_name = emb_calc.get_model_name()

    # Launch all embedding coroutines up-front
    tasks: list[asyncio.Task] = []
    task_to_item: dict[asyncio.Task, SymbolEmbeddingItem] = {}
    for it in items:
        t = asyncio.create_task(emb_calc.get_embedding(it.body, interactive=False))
        tasks.append(t)
        task_to_item[t] = it

    # As results come back, accumulate payloads and flush in batches
    pending_updates: list[tuple[ModelId, dict]] = []
    for fut in asyncio.as_completed(tasks):
        try:
            vec = await fut
        except Exception as exc:  # pragma: no cover
            item = task_to_item.get(fut)
            logger.error(
                f"Failed to compute embedding for symbol {item.symbol_id if item else 'unknown'}: {exc}",
                exc_info=True,
            )
            continue

        item = task_to_item[fut]
        pending_updates.append(
            (
                item.symbol_id,
                {"embedding_code_vec": vec, "embedding_model": model_name},
            )
        )

        if len(pending_updates) >= EMBEDDING_UPDATE_BATCH_SIZE:
            try:
                await node_repo.update(pending_updates)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                logger.error(
                    f"Batch update failed for {len(pending_updates)} embeddings: {exc}",
                    exc_info=True,
                )
            pending_updates.clear()

    # Flush remaining updates
    if pending_updates:
        try:
            await node_repo.update(pending_updates)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            logger.error(
                f"Final batch update failed for {len(pending_updates)} embeddings: {exc}",
                exc_info=True,
            )


async def schedule_missing_embeddings(pm: ProjectManager, repo: Repo) -> None:
    """Schedule embeddings for all symbols missing a vector; wait until all finish.
    Results are saved in batches as they complete."""
    emb_calc = pm.embeddings
    if not emb_calc:
        return

    node_repo = pm.data.node
    repo_id = repo.id
    PAGE_SIZE = 1_000
    offset = 0
    items: list[SymbolEmbeddingItem] = []
    while True:
        page = await node_repo.get_list(
            NodeFilter(
                repo_ids=[repo_id],
                has_embedding=False,
                limit=PAGE_SIZE,
                offset=offset,
            ),
        )
        if not page:
            break
        for sym in page:
            if sym.body:
                items.append(SymbolEmbeddingItem(symbol_id=sym.id, body=sym.body))
        offset += PAGE_SIZE
    if items:
        await schedule_symbol_embedding(node_repo, emb_calc, items)

async def schedule_outdated_embeddings(pm: ProjectManager, repo: Repo) -> None:
    """
    Re-enqueue embeddings for all symbols whose stored vector was
    generated with a *different* model than the one currently configured
    in `pm.embeddings`.
    """
    emb_calc = pm.embeddings
    if not emb_calc:  # embeddings disabled
        return

    model_name = emb_calc.get_model_name()
    node_repo = pm.data.node
    repo_id = repo.id
    PAGE_SIZE = 1_000
    offset = 0
    # TODO: Add data filter
    items: list[SymbolEmbeddingItem] = []
    while True:
        page = await node_repo.get_list(
            NodeFilter(
                repo_ids=[repo_id],
                limit=PAGE_SIZE,
                offset=offset,
            ),
        )
        if not page:
            break
        for sym in page:
            # symbol already has an embedding → but with a *different* model
            if sym.body and sym.embedding_model and sym.embedding_model != model_name:
                items.append(SymbolEmbeddingItem(symbol_id=sym.id, body=sym.body))

        offset += PAGE_SIZE
    if items:
        await schedule_symbol_embedding(node_repo, emb_calc, items)
