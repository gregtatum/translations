with open("text.txt", "r") as file:
    for line in file.readlines():
        print(f'"{line}"')
