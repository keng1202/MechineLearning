import random

input_size = 50
par_range = 20
data_size = 10000

model = [random.randint(0, par_range) for _ in range(input_size)]


fout = open("training.csv", "w")
for i in range(data_size):
    par = [random.uniform(0, par_range) for _ in range(input_size)]
    for j in range(input_size):
        fout.write(str(par[j]))
        if j != input_size-1:
            fout.write(",")
        else:
            y = sum([model[k]*par[k] for k in range(input_size)])
            fout.write("," + str(y) + "\n")
fout.close()

fout = open("testing.csv", "w")
for i in range(data_size):
    par = [random.uniform(0, par_range) for _ in range(input_size)]
    for j in range(input_size):
        fout.write(str(par[j]))
        if j != input_size-1:
            fout.write(",")
        else:
            y = sum([model[k]*par[k] for k in range(input_size)])
            fout.write("," + str(y) + "\n")
fout.close()