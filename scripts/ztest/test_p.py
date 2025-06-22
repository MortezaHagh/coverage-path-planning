class AA:
    def __init__(self, a: int = 0):
        self.__a = a
        self.__b = 1

    def get_a(self):
        return self.__a

    def get_b(self):
        return self.__b


a = AA(5)
print(a.get_a())
print(a.get_b())

a.__a = 10  # This will not change the value of __a, as it's private
print(a.get_a())  # Still prints 5, because __a is private and cannot
