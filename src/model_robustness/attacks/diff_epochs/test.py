a = ["a", "b", "c"]
b = [1, 2, 3]

for i in a:
    for j in b:
        if i == "b" and j != 1:
            continue
        print(i, j)