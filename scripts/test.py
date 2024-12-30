from dataclasses import dataclass


@dataclass
class Person():
    name: str = None
    email: int = None


Person1 = Person("Morteza")
Person1.email = 123456
print(Person1)
