from enum import Enum
from typing import Any, Dict, Optional, Type, Union

import fastapi.openapi.utils
from fastapi.openapi.constants import REF_PREFIX
from pydantic import BaseModel
from pydantic.fields import ModelField
from pydantic.schema import field_schema

orig_get_request_body = fastapi.openapi.utils.get_openapi_operation_request_body


def get_request_body_with_explode(
    *,
    body_field: Optional[ModelField],
    model_name_map: Dict[Union[Type[BaseModel], Type[Enum]], str],
) -> Optional[Dict[str, Any]]:
    """
    Patch for FastAPI to explode lists by default in OpenAPI operation request bodies.

    This function modifies the behavior of FastAPI's OpenAPI request body generation by setting the
    encoding style of array properties to "form" by default.

    Parameters:
        body_field (Optional[ModelField]): The Pydantic model field representing the request body.
        model_name_map (Dict[Union[Type[BaseModel], Type[Enum]], str]): A dictionary mapping
            Pydantic models and enums to their respective names.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the modified OpenAPI operation request
            body or None if no modifications were made.
    """
    original = orig_get_request_body(
        body_field=body_field, model_name_map=model_name_map
    )
    if not original:
        return original

    content = original.get("content", {})
    form_patch = content.get(
        "application/x-www-form-urlencoded", content.get("multipart/form-data", {})
    )

    if form_patch:
        schema_reference, schemas, _ = field_schema(
            body_field, model_name_map=model_name_map, ref_prefix=REF_PREFIX
        )
        array_props = [
            prop
            for schema in schemas.values()
            for prop, prop_schema in schema.get("properties", {}).items()
            if prop_schema.get("type") == "array"
        ]

        form_patch["encoding"] = {prop: {"style": "form"} for prop in array_props}

    return original


fastapi.openapi.utils.get_openapi_operation_request_body = get_request_body_with_explode
