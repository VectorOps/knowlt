from knowlt.stores.duckdb import DuckDBDataRepository
from knowlt.settings import ProjectSettings
from knowlt.project import ProjectManager
from knowlt.embeddings import EmbeddingWorker
from knowlt.embeddings.sentence import (
    EMBEDDINGS_EXTRA_INSTALL_HINT,
    is_sentence_transformers_available,
)
from knowlt.data import AbstractDataRepository
from knowlt import scanner
from knowlt.logger import logger


async def init_project(
    settings: ProjectSettings, refresh: bool = True
) -> ProjectManager:
    """
    Initializes the project. Settings object contains project path and/or project id.
    Then init project checks if Repo exists for the id (if provided) or absolute path.
    If it does not exist - creates a new Repo and sets that on Project instance that's returned.
    Finally, kicks off a function to recursively scan the project directory.
    """
    # TODO: Move registration out
    backend = settings.repository_backend or "duckdb"
    data: AbstractDataRepository
    if backend == "duckdb":
        data = DuckDBDataRepository(settings, db_path=settings.repository_connection)
    else:
        raise ValueError(f"Unsupported repository backend: {backend}")

    embeddings: EmbeddingWorker | None = None
    if settings.embedding and settings.embedding.enabled:
        if is_sentence_transformers_available():
            embeddings = EmbeddingWorker(
                settings.embedding.calculator_type,
                cache_backend=settings.embedding.cache_backend,
                cache_path=settings.embedding.cache_path,
                model_name=settings.embedding.model_name,
                device=settings.embedding.device,
                batch_size=settings.embedding.batch_size,
            )
        else:
            logger.warning(
                "Embeddings are enabled but optional dependencies are not installed; disabling embeddings.",
                install_hint=EMBEDDINGS_EXTRA_INSTALL_HINT,
            )

    pm = await ProjectManager.create(
        settings,
        data,
        embeddings=embeddings,
    )

    # Recursively scan the project directory and parse source files
    if refresh:
        await pm.refresh()

    return pm
