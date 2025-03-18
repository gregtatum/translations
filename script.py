from typing import TypedDict


class CasedTypedDict(TypedDict):
    def to_dict(self):
        return dict(self)


class Person(TypedDict):
    name: str
    age: int


class Employee(Person):
    job_title: str


def is_typed_dict(cls) -> bool:
    return any(
        getattr(base, "__origin__", None) is TypedDict
        for base in getattr(cls, "__orig_bases__", [])
    )


print(is_typed_dict(Person))  # True
print(is_typed_dict(Employee))  # True
print(is_typed_dict(dict))  # False
print(is_typed_dict(TypedDict))  # False
