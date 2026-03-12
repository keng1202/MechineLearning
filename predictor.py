import csv
reader = csv.reader(open("training.csv"))
input_size = 20

par = []

# par == training.csv
for row in reader:
	# unpack string row
	par.append([float(x) for x in row[:input_size+1]])

# building model, y = a1x1 + a2x2 ..... + b*1
lr = 0.00001 #learning rate
weight = [1 for x in range(input_size+1)]
epoch = 1000
# start training
for k in range(epoch):
    for i in range(len(par)):
        output = sum(weight[j]*par[i][j] for j in range(input_size)) + weight[input_size]
        err = par[i][input_size] - output
        
        # cal gradient = err*parameter, then, change weight
        gradient = []
        for j in range(input_size):
            gradient.append(err*par[i][j])
            weight[j] = weight[j] + gradient[j]*lr
        weight[input_size] = weight[input_size] + err*lr


# evaluate model
total_err = 0
for i in range(len(par)):
	output = sum(weight[j]*par[i][j] for j in range(input_size)) + weight[input_size]
	err = par[i][input_size] - output
	total_err += err**2
print("MSE:" + str(total_err/len(par)) + "\n")