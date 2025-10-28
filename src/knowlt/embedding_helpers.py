from typing import TYPE_CHECKING

from knowlt.data import NodeFilter
from knowlt.logger import logger
from knowlt.models import ModelId, Vector, Repo
from knowlt.project import ProjectManager


# Embedding helpers
async def schedule_symbol_embedding(
    node_repo, emb_calc, sym_id: ModelId, body: str
) -> None:
    # TODO: Fix me - should accumulate via queue
    vec = await emb_calc.get_embedding(body, interactive=False)

    try:
        await node_repo.update(
            sym_id,
            {
                "embedding_code_vec": vec,
                "embedding_model": emb_calc.get_model_name(),
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.error(
            f"Failed to update embedding for symbol {sym_id}: {exc}",
            exc_info=True,
        )


async def schedule_missing_embeddings(pm: ProjectManager, repo: Repo) -> None:
    """Enqueue embeddings for all symbols that still lack a vector."""
    emb_calc = pm.embeddings
    if not emb_calc:
        return

    node_repo = pm.data.node
    repo_id = repo.id
    PAGE_SIZE = 1_000
    offset = 0

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
                await schedule_symbol_embedding(
                    node_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.body,
                )
        offset += PAGE_SIZE


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
                await schedule_symbol_embedding(
                    node_repo,
                    emb_calc,
                    sym_id=sym.id,
                    body=sym.body,
                )

        offset += PAGE_SIZE
