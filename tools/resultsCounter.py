#new results counter
#record outcomes of verification and verification times

epsilons = [0.01, 0.05, 0.1, 0.5]

results = open("../results/verificationResults1l32nB.csv", "w")
line = 'index,epsilon,outcome,time\n'

for epsilon in epsilons:
    for index in range(500):
        #files
        verificationFile = '../logs/fashion1l32nB/onelayer32n' + str(epsilon) + '-' + str(index) + '.txt'
        timesFile = '../logs/fashion1l32nB/times/log1l32n-' + str(epsilon) + '-' + str(index) + '.txt'
        print(str(index) + " : " + str(epsilon))
        #read verification file
        with open(verificationFile) as f:
            data = f.read()
            #verification successful = 1
            if '  robust [=========================================================] 9/9 queries' in data:
                line += str(index) + "," + str(epsilon) + ",1,"
            #counterexample found = 0
            elif 'counterexample found' in data:
                line += str(index) + "," + str(epsilon) + ",0,"
            else:
                print("Major error; check epsilon: " + str(epsilon) + ", at index: " + str(index))
            f.close()

        #read times file
        with open(timesFile) as f:
            data = f.read()
            time = data.replace('Runtime on 1l32n-' + str(epsilon) + '-image:' + str(index) + ': ', '')
            line += time
            f.close()
print(line)
results.write(line)
results.close()