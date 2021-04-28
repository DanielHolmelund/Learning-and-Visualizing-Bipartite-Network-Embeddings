def loader(text_file):
    """
    :param text_file:
    :return:
    """
    f = open(text_file, "r")
    data = f.read()
    data = data.split("\n")
    lenght = len(data)
    U, V, values = [], [], []
    for i in range(lenght):
        #data = data[i].split(" ")
        U.append(int(data[i].split(" ")[0]))
        V.append(int(data[i].split(" ")[1]))
        values.append(int(data[i].split(" ")[2]))
    U, V, values = torch.tensor(U), torch.tensor(V), torch.tensor(values)
    return U, V, values

#U, V, values = loader(text_file)