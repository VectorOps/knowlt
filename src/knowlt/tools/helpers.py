import datetime
import decimal
import json
from pydantic import BaseModel
from typing import Any
from enum import Enum


def default(o: Any):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def json_dumps(val: Any) -> str:
    try:
        return json.dumps(
            val,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=default,
        )
    except Exception:
        return str(val)


def stringify(value: Any) -> str:
    # Normalize nested collections to JSON; simple scalars to str
    if isinstance(value, (dict, list, tuple, set)):
        s = json_dumps(value)
    else:
        s = str(value)
    # Normalize newlines
    return s.replace("\r\n", "\n").replace("\r", "\n")


def convert_to_python(obj: Any) -> Any:
    """
    Recursively turn Pydantic models / Enums / collections into
    plain-Python (JSON-serialisable) structures.
    """

    if isinstance(obj, BaseModel):
        return obj.model_dump(by_alias=True, exclude_none=True)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return [convert_to_python(v) for v in obj]
    if isinstance(obj, set):
        return sorted(convert_to_python(v) for v in obj)
    return obj
