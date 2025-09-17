class AA:
    def __init__(self, a: int = 0):
        self.a = 1


all_aas = [AA(i) for i in range(5)]

for a in all_aas:
    a.a = 2


for a in all_aas:
    print(a.a)
