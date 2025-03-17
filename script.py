from enum import Enum
from typing import NamedTuple, Any, Protocol, TypeVar

Cls = TypeVar("Cls", bound=NamedTuple)

class CasedNamedTupleProtocol(Protocol):
    @classmethod
    def from_dict(cls: type[Cls], data: dict[str, Any]) -> Cls: ...
    
    def to_dict(self) -> dict[str, Any]: ...
    
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
        cls_type: type[Cls],
        casing: Casing = Casing.underscore,
        kebab: list[str] = [],
        underscore: list[str] = [],
    ):
        """Register casing details for a NamedTuple class."""
        CasingManager.kebab_cased_fields_by_class[cls_type] = [v.replace("-", "_") for v in kebab]
        CasingManager.underscore_cased_fields_by_class[cls_type] = [v.replace("-", "_") for v in underscore]
        CasingManager.casing_by_class[cls_type] = casing

    @staticmethod
    def get_field_casing(cls_type: type, key_name: str):
        """
        Determines the correct casing for a given field name based on class configuration.
        """
        key_name = key_name.replace("-", "_")
        casing = CasingManager.casing_by_class.get(cls_type, Casing.underscore)

        if casing is Casing.underscore:
            kebab_fields = CasingManager.kebab_cased_fields_by_class.get(cls_type, [])
            return Casing.kebab if key_name in kebab_fields else Casing.underscore

        underscore_fields = CasingManager.underscore_cased_fields_by_class.get(cls_type, [])
        return Casing.underscore if key_name in underscore_fields else Casing.kebab


def casing(
    casing: Casing = Casing.underscore,
    kebab: list[str] = [],
    underscore: list[str] = [],
):
    """
    Decorator to configure casing settings for a NamedTuple.
    """
    def wrapper(cls: type[Cls]) -> type[Cls]:
        CasingManager.add_class(cls, casing, kebab, underscore)

        def from_dict(cls, data: dict[str, Any]) -> Cls:
            """Convert a dict with kebab-case keys to an instance of the NamedTuple."""
            normalized_data = {}

            for key, value in data.items():
                # Normalize key names for internal use
                normalized_key = key.replace("-", "_")

                # Ensure normalized key exists in class definition
                if normalized_key not in cls._fields:
                    raise KeyError(f"Unexpected field '{key}' in input data.")  # Show original key

                normalized_data[normalized_key] = value

            return cls(**normalized_data)

        def to_dict(self) -> dict[str, Any]:
            """Convert the NamedTuple instance back to a dict with kebab-case keys."""
            cls_type = type(self)
            serialized_data = {}

            for field in self._fields:
                field_casing = CasingManager.get_field_casing(cls_type, field)
                serialized_key = field.replace("_", "-") if field_casing == Casing.kebab else field
                serialized_data[serialized_key] = getattr(self, field)

            return serialized_data

        setattr(cls, "from_dict", classmethod(from_dict))
        setattr(cls, "to_dict", to_dict)
        return cls

    return wrapper


@casing(kebab=["kebab-key"])
class MyData(NamedTuple):
    underscore_key: int
    kebab_key: str  # Ensure correct field name (no typo)

# Example Usage
data = {"underscore_key": 42, "kebab-key": "value"}  # Kebab case input
obj = MyData.from_dict(data)
print("MyData", obj)  # MyData(underscore_key=42, kebab_key='value')

round_trip = obj.to_dict()
print("Round trip", round_trip)  # {'underscore_key': 42, 'kebab-key': 'value'}
