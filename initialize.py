import random

input_size = 500
par_range = 1000

model = [random.randint(-par_range, par_range) for _ in range(input_size)]

fout = open("model.txt", "w")
for i in range(input_size):
    fout.write(str(model[i]) + "\n")
fout.close()

fout = open("training.csv", "w")
for i in range(input_size):
    par = [random.uniform(-par_range, par_range) for _ in range(input_size)]
    for j in range(input_size):
        fout.write(str(par[j]))
        if j != input_size-1:
            fout.write(",")
        else:
            y = sum([model[k]*par[k] for k in range(input_size)])
            fout.write("," + str(y) + "\n")
fout.close()

fout = open("testing.csv", "w")
fout2 = open("ans.csv", "w")
for i in range(input_size):
    par = [random.uniform(-par_range, par_range) for _ in range(input_size)]
    for j in range(input_size):
        fout.write(str(par[j]))
        if j != input_size-1:
            fout.write(",")
        else:
            y = sum([model[k]*par[k] for k in range(input_size)])
            fout.write("\n")
            fout2.write(str(y) + "\n")
fout.close()
fout2.close()