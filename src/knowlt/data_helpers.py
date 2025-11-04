from typing import List, Dict

from knowlt.data import AbstractNodeRepository, NodeFilter, AbstractPackageRepository
from knowlt.models import ModelId, Node, NodeKind, File, Package


# Helpers
def resolve_node_hierarchy(symbols: List[Node]) -> None:
    """
    Populate in-memory parent/child links inside *symbols* **in-place**.

    • parent_ref   ↔  points to the parent Node instance
    • children     ↔  list with direct child Node instances

    Function is no-op when list is empty.
    """
    if not symbols:
        return

    id_map: dict[ModelId | None, Node] = {s.id: s for s in symbols if s.id}
    # clear any previous links to avoid duplicates on repeated invocations
    for s in symbols:
        s.children.clear()
        s.parent_ref = None

    for s in symbols:
        pid = s.parent_node_id
        if pid and (parent := id_map.get(pid)):
            s.parent_ref = parent
            parent.children.append(s)


async def include_parents(
    repo: AbstractNodeRepository,
    symbols: List[Node],
) -> List[Node]:
    """
    Traverse all symbol parents and include them to in the tree.
    """
    source = symbols

    while source:
        parent_ids = {s.parent_node_id for s in source if s.parent_node_id}
        if not parent_ids:
            break

        all_parents = await repo.get_list_by_ids(list(parent_ids))
        parents = {p.id: p for p in all_parents if p.kind != NodeKind.LITERAL}

        for s in source:
            if s.parent_node_id:
                parent = parents.get(s.parent_node_id)
                if not parent:
                    continue
                s.parent_ref = parent

                for i, c in enumerate(parent.children):
                    if c.id == s.id:
                        parent.children[i] = s
                        break
                else:
                    parent.children.append(s)

        source = list(parents.values())

    return symbols


async def include_direct_descendants(
    repo: AbstractNodeRepository,
    symbols: List[Node],
) -> List[Node]:
    """
    Ensure every symbol in *symbols* has its direct descendants attached.
    After resolving the hierarchy, any symbol that became a child of another
    returned symbol is dropped from the top-level list (to avoid duplicates).
    The original order of the parent symbols is preserved.
    """
    if not symbols:
        return symbols

    parent_ids = [s.id for s in symbols if s.id and s.kind != NodeKind.LITERAL]
    if parent_ids:
        children = await repo.get_list(NodeFilter(parent_ids=parent_ids))
        seen_ids = {s.id for s in symbols}
        for c in children:
            if c.id not in seen_ids:
                symbols.append(c)
                seen_ids.add(c.id)

    resolve_node_hierarchy(symbols)

    parent_id_set = set(parent_ids)
    result = [s for s in symbols if s.parent_node_id not in parent_id_set]

    result = include_parents(repo, result)

    return result


async def post_process_search_results(
    repo: AbstractNodeRepository,
    results: List[Node],
    limit: int,
) -> List[Node]:
    """
    Post-processes a list of search results to improve relevance.

    - Filters out parent nodes if their children are also in the results.
    - Enforces the final limit on the primary results.
    - Includes direct descendants of the final results.
    - Designed to allow for future inclusion of reranking models.
    """
    # Filter out nodes that are parents of other nodes in the result set
    all_ids = {n.id for n in results}
    parent_ids_in_results = {
        n.parent_node_id for n in results if n.parent_node_id in all_ids
    }

    processed_results = [n for n in results if n.id not in parent_ids_in_results]

    # TODO: Add reranking logic here in the future.

    # Limit the number of primary results
    final_results = processed_results[:limit]

    # Enrich with direct descendants
    final_results = await include_direct_descendants(repo, final_results)

    return final_results


# Files/Packages helpers
async def populate_packages_for_files(
    package_repo: AbstractPackageRepository, files: List[File]
) -> Dict[ModelId, Package]:
    """
    Populate the `package` attribute on each File in-place by batch-loading
    referenced packages. Returns a map of package_id -> Package for reuse.
    """
    if not files:
        return {}

    pkg_ids = {f.package_id for f in files if getattr(f, "package_id", None)}
    if not pkg_ids:
        return {}

    packages = await package_repo.get_by_ids(list(pkg_ids))
    package_by_id: Dict[ModelId, Package] = {p.id: p for p in packages}

    for f in files:
        if f.package_id and f.package_id in package_by_id:
            f.package = package_by_id[f.package_id]

    return package_by_id
