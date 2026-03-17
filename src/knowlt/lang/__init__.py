from importlib import import_module


def _safe_import(module_name: str) -> None:
    try:
        import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError:
        return


for _module_name in (
    "c",
    "cpp",
    "python",
    "text",
    "markdown",
    "golang",
    "typescript",
    "javascript",
    "terraform",
):
    _safe_import(_module_name)
