#read text from .txt file
#check if text contains counterexample found or not
#get time to verify
#loop through indecies in the txt filename
#do epsilon 0.01,0.05,0.1,0.5
#write to csv

#eps=0.01
epsilon = 0.01
csv01 = open("verificationResults" + str(epsilon) + ".csv", "w")
line = ''
for i in range(0,500,1):
    verificationFile = './logs/fashion1l32n/onelayer32n' + str(epsilon) + '-' + str(i) + '.txt'
    timesFile = './logs/times/log1l32n-' + str(epsilon) + '-' + str(i) + '.txt'
    
    #read verification file
    with open(verificationFile) as f:
        data = f.read()
        #verification successful = 1
        if '  robust [=========================================================] 9/9 queries' in data:
            line += str(i) + ",1,"
        #counterexample found = 0
        elif 'counterexample found' in data:
            line += str(i) + ",0,"
        else:
            print("Major error; check " + str(epsilon) + " " + str(i))
        f.close()

    #read times file
    with open(timesFile) as f:
        data = f.read()
        time = data.replace('Runtime on 1l32n-' + str(epsilon) + '-image:' + str(i) + ': ', '')
        line += time
        f.close()

print(line)
csv01.write(line)
csv01.close()

#eps=0.05
epsilon = 0.05
csv05 = open("verificationResults" + str(epsilon) + ".csv", "w")
line = ''
for i in range(0,500,1):
    verificationFile = './logs/fashion1l32n/onelayer32n' + str(epsilon) + '-' + str(i) + '.txt'
    timesFile = './logs/times/log1l32n-' + str(epsilon) + '-' + str(i) + '.txt'
    
    #read verification file
    with open(verificationFile) as f:
        data = f.read()
        #verification successful = 1
        if '  robust [=========================================================] 9/9 queries' in data:
            line += str(i) + ",1,"
        #counterexample found = 0
        elif 'counterexample found' in data:
            line += str(i) + ",0,"
        else:
            print("Major error; check " + str(epsilon) + " " + str(i))
        f.close()

    #read times file
    with open(timesFile) as f:
        data = f.read()
        time = data.replace('Runtime on 1l32n-' + str(epsilon) + '-image:' + str(i) + ': ', '')
        line += time
        f.close()

print(line)
csv05.write(line)
csv05.close()

#eps=0.1
epsilon = 0.1
csv1 = open("verificationResults" + str(epsilon) + ".csv", "w")
line = ''
for i in range(0,500,1):
    verificationFile = './logs/fashion1l32n/onelayer32n' + str(epsilon) + '-' + str(i) + '.txt'
    timesFile = './logs/times/log1l32n-' + str(epsilon) + '-' + str(i) + '.txt'
    
    #read verification file
    with open(verificationFile) as f:
        data = f.read()
        #verification successful = 1
        if '  robust [=========================================================] 9/9 queries' in data:
            line += str(i) + ",1,"
        #counterexample found = 0
        elif 'counterexample found' in data:
            line += str(i) + ",0,"
        else:
            print("Major error; check " + str(epsilon) + " " + str(i))
        f.close()

    #read times file
    with open(timesFile) as f:
        data = f.read()
        time = data.replace('Runtime on 1l32n-' + str(epsilon) + '-image:' + str(i) + ': ', '')
        line += time
        f.close()

print(line)
csv1.write(line)
csv1.close()

#eps=0.5
epsilon = 0.5
csv5 = open("verificationResults" + str(epsilon) + ".csv", "w")
line = ''
for i in range(0,500,1):
    verificationFile = './logs/fashion1l32n/onelayer32n' + str(epsilon) + '-' + str(i) + '.txt'
    timesFile = './logs/times/log1l32n-' + str(epsilon) + '-' + str(i) + '.txt'
    
    #read verification file
    with open(verificationFile) as f:
        data = f.read()
        #verification successful = 1
        if '  robust [=========================================================] 9/9 queries' in data:
            line += str(i) + ",1,"
        #counterexample found = 0
        elif 'counterexample found' in data:
            line += str(i) + ",0,"
        else:
            print("Major error; check " + str(epsilon) + " " + str(i))
        f.close()

    #read times file
    with open(timesFile) as f:
        data = f.read()
        time = data.replace('Runtime on 1l32n-' + str(epsilon) + '-image:' + str(i) + ': ', '')
        line += time
        f.close()

print(line)
csv5.write(line)
csv5.close()