from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List, Any, Type, Set
from . import settings


# Various settings parsing helpers helpers
class CliOption(BaseModel):
    """Represents a single command-line option."""

    flag: str
    aliases: List[str] = Field(default_factory=list)
    description: str
    is_required: bool
    default_value: Any
    is_group: bool = False


def load_settings(
    cli: bool = False,
    env_prefix: Optional[str] = None,
    env_file: Optional[str] = None,
    toml_file: Optional[str] = None,
    json_file: Optional[str] = None,
    **kwargs,
) -> settings.ProjectSettings:
    config_dict = SettingsConfigDict(
        cli_parse_args=cli,
        env_prefix=env_prefix or "",
        env_file=env_file,
        toml_file=toml_file,
        json_file=json_file,
    )

    class Settings(settings.ProjectSettings):
        model_config = config_dict

    return Settings(**kwargs)


def iter_settings(
    model: Type[BaseModel],
    *,
    kebab: bool = False,
    implicit_flags: Optional[bool] = None,
) -> List[CliOption]:
    """
    Return a list of `CliOption` models for every CLI option that *model* would accept.

    Parameters
    ----------
    model : BaseModel | BaseSettings subclass
    kebab : convert snake_case to kebab‑case (matches ``cli_kebab_case``)
    implicit_flags : add ``--no-flag`` for bools when ``cli_implicit_flags`` would be true
                     (pass None to autodetect from model.model_config)
    """
    seen: Set[str] = set()
    out: List[CliOption] = []

    # Resolve whether booleans get --no-* automatically
    if implicit_flags is None:
        # pydantic-settings defaults to True for cli_implicit_flags
        implicit_flags = bool(
            getattr(model, "model_config", {}).get("cli_implicit_flags", True)
        )

    def add(option: CliOption) -> None:
        if option.flag not in seen:
            seen.add(option.flag)
            out.append(option)

    def _walk(cls: Type[BaseModel], dotted: str = "") -> None:
        for name, field in cls.model_fields.items():
            path = f"{dotted}.{name}" if dotted else name
            desc = field.description or ""
            ann = field.annotation

            # Determine if the field is a nested model for recursion
            is_nested_model = False
            nested_types_for_recursion = []
            if hasattr(ann, "__pydantic_generic_metadata__"):  # parametrised generics
                metadata = getattr(ann, "__pydantic_generic_metadata__")
                args = metadata.get("args")
                if args:
                    for arg in args:
                        if isinstance(arg, type) and issubclass(arg, BaseModel):
                            is_nested_model = True
                            nested_types_for_recursion.append(arg)
            elif isinstance(ann, type) and issubclass(ann, BaseModel):
                is_nested_model = True
                nested_types_for_recursion.append(ann)

            # Flag generation logic
            all_flags = []
            path_prefix = (
                ".".join(p.replace("_", "-") for p in dotted.split("."))
                if kebab and dotted
                else ""
            )
            path_prefix_dot = f"{path_prefix}." if path_prefix else ""

            choices = []
            if field.validation_alias:
                if isinstance(field.validation_alias, str):
                    choices.append(field.validation_alias)
                elif hasattr(field.validation_alias, "choices"):  # AliasChoices
                    choices.extend(map(str, field.validation_alias.choices))

            flag_name = (
                ".".join(p.replace("_", "-") for p in path.split("."))
                if kebab
                else path
            )
            if choices:
                for choice in choices:
                    if len(choice) == 1 and not path_prefix:
                        all_flags.append(f"-{choice}")
                    else:
                        full_name = f"{path_prefix_dot}{choice}"
                        all_flags.append(f"--{full_name}")
            else:
                all_flags.append(f"--{flag_name}")

            all_flags.sort(key=lambda x: (x.startswith("--"), len(x)), reverse=True)
            main_flag = all_flags[0]
            aliases = all_flags[1:]

            add(
                CliOption(
                    flag=main_flag,
                    aliases=sorted(aliases),
                    description=desc,
                    is_required=field.is_required(),
                    default_value=(
                        field.get_default() if not field.is_required() else ...
                    ),
                    is_group=is_nested_model,
                )
            )

            # negated boolean
            if implicit_flags and field.annotation is bool:
                add(
                    CliOption(
                        flag=f"--no-{flag_name}",
                        aliases=[],
                        description=f"Disable '{flag_name}'",
                        is_required=False,
                        default_value=...,
                    )
                )

            # Recurse into nested models
            if is_nested_model:
                for nested_type in nested_types_for_recursion:
                    _walk(nested_type, path)

    _walk(model)
    return sorted(out, key=lambda o: o.flag)


def print_help(model: Type[BaseModel], script_name: str, kebab: bool = True):
    """
    Print a formatted help message for a Pydantic settings model.

    Parameters
    ----------
    model : BaseModel | BaseSettings subclass
    script_name : file name to show in the "usage: " line
    kebab : convert snake_case to kebab‑case for CLI flags
    """
    print(f"usage: {script_name} [OPTIONS]")
    print("\nOptions:")
    for opt in iter_settings(model, kebab=kebab):
        flags = [opt.flag] + opt.aliases
        flag_str = ", ".join(flags)
        line = f"  {flag_str:<40} {opt.description}"

        details = []
        if opt.is_required:
            details.append("required")

        if not opt.is_group and opt.default_value is not ...:
            # for multiline defaults, only show the first line
            default_str = str(opt.default_value).split("\n")[0]
            details.append(f"default: {default_str!r}")

        if details:
            line += f" [{', '.join(details)}]"
        print(line)
