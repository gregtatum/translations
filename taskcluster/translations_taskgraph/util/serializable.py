"""
This file contains helpers for converting untyped JSON configurations into typed JSON.

1. Dataclasses require underscores, while our configs uses kebab casing. These helpers
   handle the serialization and deserialization to change underscores and dashes.
   See the KebabDataclass.

2. The pipeline uses two different validation schemes, voluptuous types, and json schema.
   These helpers have convertors for both validation types so the source of truth is
   the type-safe dataclass, but then stored and generated yaml files can still be
   validated.

3. Dataclasses don't validate unions of string literals at runtime. These helpers do.
"""

from enum import Enum
import types
from typing import (
    Optional,
    Any,
    Literal,
    Tuple,
    Union,
    cast,
    get_origin,
    get_args,
    Type,
    TypeVar,
    get_type_hints,
)
import voluptuous

TypedDictCls = TypeVar("TypedDictCls")


class Casing(Enum):
    kebab = "kebab"
    underscore = "underscore"


class CasingManager:
    """
    Static class to manage cases
    """

    kebab_cased_fields_by_class: dict[type, list[str]] = {}
    underscore_cased_fields_by_class: dict[type, list[str]] = {}
    casing_by_class: dict[type, Casing] = {}

    @staticmethod
    def add_class(
        cls_type: type[TypedDictCls],
        casing: Casing = Casing.underscore,
        kebab: list[str] = [],
        underscore: list[str] = [],
    ):
        # Store them all as underscores.
        CasingManager.kebab_cased_fields_by_class[cls_type] = [v.replace("-", "_") for v in kebab]
        CasingManager.underscore_cased_fields_by_class[cls_type] = [
            v.replace("-", "_") for v in underscore
        ]
        CasingManager.casing_by_class[cls_type] = casing

    @staticmethod
    def get_field_casing(cls_type: type, key_name: str):
        """
        Apply the casing of the class, and the casing of the fields to determine the
        individual casing of a field.
        """
        key_name = key_name.replace("-", "_")
        casing = CasingManager.casing_by_class.get(cls_type, Casing.underscore)
        if casing is Casing.underscore:
            # Casing is underscore by default.
            kebab_fields = CasingManager.kebab_cased_fields_by_class.get(cls_type, None)
            if kebab_fields and key_name in kebab_fields:
                return Casing.kebab
            return Casing.underscore

        # Casing is kebab by default.
        underscore_fields = CasingManager.underscore_cased_fields_by_class.get(cls_type, None)
        if underscore_fields and key_name in underscore_fields:
            return Casing.underscore
        return Casing.kebab


def has_type_annotations(cls: type):
    return bool(getattr(cls, "__annotations_", None))


def get_fields(data_type: type) -> dict[str, type]:
    return data_type.__annotations__


def _deserialize(data_type: type, data: Any, key_path: str):
    """
    Recursively deserialize the dataclasses in a stricter fashion. This handles the
    kebab casing requirements for the KebabDataclass. It also validates Unions
    of Literals.
    """
    assert not is_type_optional(
        data_type
    ), "Optional types should not be passed into this function."

    if data_type is Any:
        # Do not type check any types, just pass them through.
        return data

    key_path_display = key_path or "<root>"
    import sys

    sys.stdout = sys.__stdout__

    if has_type_annotations(data_type):
        if not isinstance(data, dict):
            print("Data:", data)
            raise ValueError(f'Expected a dictionary at "{key_path_display}".')

        fields: dict[str, type] = get_fields(data_type)
        required_fields = {
            field_name
            for field_name, field_type in fields.items()
            if not is_type_optional(field_type)
        }

        vargs = {}
        for dict_key, dict_value in data.items():
            dict_value: Any = dict_value
            if not isinstance(dict_key, str):
                print(dict_key)  # type: ignore[reportUnknownArgumentType]
                raise ValueError(f'Expected a dict key to be a string at "{key_path_display}".')
            # Replace the kebab casing where needed.
            vargs_key = cast(str, dict_key)  # type: ignore[reportUnnecessaryCast]
            if CasingManager.get_field_casing(data_type, dict_key) is Casing.kebab:
                vargs_key = dict_key.replace("-", "_")

            next_key_path = f"{key_path}.{dict_key}" if key_path else dict_key
            if vargs_key not in fields:
                underscore_vargs_key = vargs_key.replace("-", "_")
                kebab_vargs_key = vargs_key.replace("_", "-")
                extra_message = ""
                if underscore_vargs_key in fields:
                    extra_message = f' The underscore version of this field was found "{underscore_vargs_key}". Do you need to update the type definition?'
                if kebab_vargs_key in fields:
                    extra_message = f' The kebab version of this field was found "{kebab_vargs_key}". Do you need to update the type definition?'

                raise ValueError(
                    f'Unexpected field "{next_key_path}" when deserializing dataclass "{data_type.__name__}".{extra_message}'
                )

            field_type = fields[vargs_key]
            field_type = extract_optional_properties(field_type)[1]

            if has_type_annotations(field_type):
                vargs[vargs_key] = from_dict(field_type, dict_value, next_key_path)
            else:
                vargs[vargs_key] = _deserialize(field_type, dict_value, next_key_path)
            required_fields.discard(vargs_key)

        if required_fields:
            if len(required_fields) == 1:
                raise ValueError(
                    f'Field "{key_path}.{next(iter(required_fields))}" missing when deserializing dataclass "{data_type.__name__}".'
                )
            else:
                raise ValueError(
                    f'Fields missing when deserializing "{data_type.__name__}" at "{key_path_display}": {", ".join(required_fields)}.'
                )

        return data_type(**vargs)

    type_origin = get_origin(data_type)

    if type_origin is list:
        print("!!! list")
        if not isinstance(data, list):
            print(data)
            raise ValueError(f'A list was not provided at "{key_path_display}"')
        list_type = get_args(data_type)[0]
        return [
            _deserialize(list_type, list_item, f"{key_path}[{index}]")
            for index, list_item in enumerate(data)  # type: ignore[reportUnknownArgumentType]
        ]

    primitives = {float, int, str, dict, None}

    if type_origin is Union:
        print("!!! Union")
        union_items = get_args(data_type)
        literals: Optional[list[str]] = None
        has_literals = False
        is_all_literals = True
        for union_item in union_items:
            if get_origin(union_item) is Literal:
                has_literals = True
            else:
                is_all_literals = False

        if has_literals:
            if is_all_literals:
                literals = [get_args(union_item)[0] for union_item in union_items]
                if data in literals:
                    return data
                else:
                    raise ValueError(
                        f'An unexpected value "{data}" was provided at "{key_path_display}", expected it to be one of: {literals}'
                    )
            else:
                raise TypeError(
                    f'Union contained a mix of literal and non-literal types at "{key_path_display}"'
                )

        for union_item in union_items:
            if union_item not in primitives:
                raise TypeError(
                    f'A union contained a non-primitive value "{union_item}" at "{key_path_display}"'
                )

        return data

    if type_origin not in primitives:
        raise ValueError(f'Non-primitive value {type_origin} provided at "{key_path_display}".')

    print("!!! primitive")

    return data


def is_named_tuple(obj: Any) -> bool:
    return isinstance(obj, tuple) and hasattr(obj, "_fields")  # type: ignore[reportUnknownArgumentType]


def from_dict(cls: Type[TypedDictCls], data: Any, key_path: str = "") -> TypedDictCls:
    if not isinstance(data, dict):
        raise ValueError("Expected a dict")
    result = _deserialize(cls, data, key_path)
    assert is_named_tuple(result)
    return result  # type: ignore[This runtime check fails]


def _serialize(value: Any) -> Any:
    if isinstance(value, (float, int, str, dict, type(None))):
        return value

    if isinstance(value, list):
        return [_serialize(v) for v in value]

    if is_named_tuple(value):
        result = {}
        for field_name in get_fields(value):
            # Handle the Kebab casing.
            value_type: type = type(value)
            serialized_field_name = field_name
            if CasingManager.get_field_casing(value_type, field_name) == Casing.kebab:
                serialized_field_name = field_name.replace("_", "-")

            # Serialize the value, but omit it if the value is None.
            serialized_value = _serialize(getattr(value, field_name))
            if serialized_value is not None:
                result[serialized_field_name] = serialized_value

        return result

    raise ValueError("Unexpected value type")


def to_dict(typed_dict: Any) -> dict[str, Any]:
    return _serialize(typed_dict)


def casing(
    casing: Casing = Casing.underscore,
    kebab: list[str] = [],
    underscore: list[str] = [],
):
    """
    Decorator to instantiate a stricter dataclass with the given configuration.
    """

    def wrapper(cls: type[TypedDictCls]) -> type[TypedDictCls]:
        CasingManager.add_class(cls, casing, kebab, underscore)

        # Get type hints to check for Optional fields
        type_hints = get_type_hints(cls)

        # Modify class attributes to default to None if they are Optional
        for attr, attr_type in type_hints.items():
            if is_type_optional(attr_type) and not hasattr(cls, attr):
                setattr(cls, attr, None)

        return cls

    return wrapper


def extract_voluptuous_optional_type(
    t: Any,
) -> Tuple[type[voluptuous.Required] | type[voluptuous.Optional], type]:
    """
    Determines if a property is optional or not.
    Optional[T] is an alias for Union[T, None].
    """
    origin = get_origin(t)
    if origin is not Union:
        return voluptuous.Required, t

    args = get_args(t)
    if len(args) != 2:
        return voluptuous.Required, t

    arg1, arg2 = args
    if arg1 is types.NoneType:
        return voluptuous.Optional, arg2

    if arg2 is types.NoneType:
        return voluptuous.Optional, arg1

    return voluptuous.Required, t


def extract_optional_properties(
    t: Any,
) -> tuple[bool, Any]:
    """
    Determines if a property is optional or required. If it is required it adds it
    to the required list.

    The following values are returned:
        Optional[T] returns T
        T return T

    Optional[T] is an alias for Union[T, None].
    """
    origin = get_origin(t)
    if origin is not Union:
        return False, t

    args = get_args(t)

    if len(args) == 1:
        return False, t

    if len(args) == 2:
        # Extract the non-optional parameter.
        arg1, arg2 = args
        if arg1 is types.NoneType:
            return True, arg2

        if arg2 is types.NoneType:
            return True, arg1

    # This is an optional Union, for instance an optional Union of string literals.
    args_no_none = [arg for arg in args if arg is not types.NoneType]

    if len(args_no_none) == len(args):
        return False, args

    return True, Union[tuple(args_no_none)]


def is_type_optional(t: type) -> bool:
    return extract_optional_properties(t)[0]


def build_voluptuous_schema(
    value: type,
    key_requirement: type[voluptuous.Required] | type[voluptuous.Optional] = voluptuous.Required,
):
    # Is this just a basic data type?
    if value in [str, float, int, bool]:
        if key_requirement == voluptuous.Optional:
            # The validation will fail if the value `null` is passed in for a key rather
            # than just omitting the key. Handle that case here by allowing null.
            return voluptuous.Any(value, None)
        else:
            return value

    # Handle creating the schema from a dataclass. Also handle the kebab casing.
    if is_named_tuple(value):
        schema_dict = {}
        for field_name, field_type in get_fields(value).items():
            key_wrapper, field_type = extract_voluptuous_optional_type(field_type)

            if CasingManager.get_field_casing(value, field_name) is Casing.kebab:
                field_name = field_name.replace("_", "-")
            else:
                field_name = field_name

            schema_dict[key_wrapper(field_name)] = build_voluptuous_schema(field_type, key_wrapper)
        return schema_dict

    origin = get_origin(value)

    # Convert a dict type to voluptous
    # dict[str, str] => {str: str}
    if origin is dict:
        args = get_args(value)
        key: Any = voluptuous.Any
        dict_value = voluptuous.Any

        if len(args) > 0:
            key = build_voluptuous_schema(args[0])

        if len(args) == 2:
            dict_value = build_voluptuous_schema(args[1])

        return {key: dict_value}

    # Convert a list type to voluptous
    # list[float] => [float]
    # list[int]   => [int]
    # list        => list
    if origin is list:
        args = get_args(origin)
        if not args:
            return list
        return [build_voluptuous_schema(args[0])]

    if origin is Union:
        args = get_args(value)
        enum = []
        for literal_type in args:
            if get_origin(literal_type) is not Literal:
                print(literal_type)
                raise ValueError("Currently only Literals are currently supported in Unions")
            literal_str = get_args(literal_type)[0]
            assert isinstance(literal_str, str), "The literal is a string"
            enum.append(literal_str)

        return voluptuous.In(enum)

    raise Exception("Unknown type when converting from dataclass to voluptuous")


json_schema_type = {str: "string", float: "number", int: "number", bool: "boolean"}


def build_json_schema(type_value: type, is_key_optional: bool = False):
    """
    https://json-schema.org/
    """

    # Is this just a basic data type?
    if type_value in json_schema_type:
        if is_key_optional:
            # The validation will fail if the value `null` is passed in for a key rather
            # than just omitting the key. Handle that case here by allowing null.
            return {"type": [json_schema_type[type_value], "null"]}
        else:
            return {"type": json_schema_type[type_value]}

    # Handle creating the schema from a dataclass. Also handle the kebab casing.
    if is_named_tuple(type_value):
        properties: dict[str, Any] = {}
        schema_dict: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        }
        required: list[str] = []
        for field_name, field_type in get_fields(type_value).items():
            is_optional, field_type = extract_optional_properties(field_type)

            if CasingManager.get_field_casing(type_value, field_name) is Casing.kebab:
                field_name = field_name.replace("_", "-")
            else:
                field_name = field_name

            if not is_optional:
                required.append(field_name)
            properties[field_name] = build_json_schema(field_type, is_optional)
        if required:
            schema_dict["required"] = required
        return schema_dict

    origin = get_origin(type_value)

    if origin is dict:
        args = get_args(type_value)
        additional_properties = {}
        if len(args) == 2:
            additional_properties = build_json_schema(args[1])

        return {"type": "object", "additionalProperties": additional_properties}

    if origin is list:
        args = get_args(origin)
        if not args:
            return {
                "type": "array",
                "items": {},
            }

        return (
            {
                "type": "array",
                "items": build_json_schema(args[0]),
            },
        )

    # Unions are a bit more complicated to handle. In our case make the assumption that
    # only Literals are allowed as a union, as these can than be checked by the schema
    # as an enum type.
    if origin is Union:
        # Only literal args are supported here.
        args = get_args(type_value)
        enum = []

        for literal_type in args:
            if get_origin(literal_type) is not Literal:
                print(literal_type)
                raise ValueError("Currently only Literals are currently supported in Unions")
            literal_str = get_args(literal_type)[0]
            assert isinstance(literal_str, str), "The literal is a string"
            enum.append(literal_str)

        return {
            "type": "string",
            "enum": enum,
        }

    raise Exception("Unknown type when converting from dataclass to voluptuous")
