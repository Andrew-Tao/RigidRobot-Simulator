from typing import TYPE_CHECKING, TypeVar, Generic, List, Optional, Union,Any
from typing import Protocol


T = TypeVar("T", int, float, str)
# print(T)
# T = int
# print(T)

def add(a: Union[int, float], b: int) -> Union[int, float]:
    return (a + b) 

class A(Generic[T]):
    def __init__(self, a: Any) -> None:
        self.a = a 
    def add(self, b: Any) -> Any:
        return (self.a + b)

class B(Generic[T]):
    def __init__(self, a: Any) -> None:
        self.a = a 
    def minus (self, b: Any) -> Any:
        return (self.a + b)
class AddProtocal(Protocol):
    def add(self, *arg, **kwargs) -> Any:
        ...

def myfunction(obj: AddProtocal, b: Any):
    return 0

object_A = A(5.0)
object_B = B(4.0)

myfunction(object_A, 4.0)





# a = 5.0
# b = 5 
# c = add(a, b)


result = object_A.add(5)

print(result)
# print(c)
# print(type(c))

