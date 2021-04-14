import pickle


#Display data to gather insight
filename = 'ind.citeseer.tx'
x = pickle.load(open(filename, 'rb'), encoding="latin1")

print(x)