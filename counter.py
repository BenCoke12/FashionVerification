#read text from .txt file
#check if text contains counterexample found or not
#loop through indecies 0-99 in the txt filename
#do epsilon 0.01,0.05,0.1,0.5
#write to csv

#eps=0.01
csv01 = open("moreresults0.01.csv", "w")
for i in range(100,500,1):
    filename = './logs/fashion1l32n/onelayer32n0.01-' + str(i) + '.txt'

    with open(filename) as f:
        data = f.read()
        if '  robust [=========================================================] 9/9 queries' in data:
            csv01.write(str(i) + ",1\n")
        elif 'counterexample found' in data:
            csv01.write(str(i) + ",0\n")
        else:
            print("Major error; check" + filename)
        f.close()
csv01.close()

#eps=0.05
csv05 = open("moreresults0.05.csv", "w")
for i in range(100,500,1):
    filename = './logs/fashion1l32n/onelayer32n0.05-' + str(i) + '.txt'

    with open(filename) as f:
        data = f.read()
        if '  robust [=========================================================] 9/9 queries' in data:
            csv05.write(str(i) + ",1\n")
        elif 'counterexample found' in data:
            csv05.write(str(i) + ",0\n")
        else:
            print("Major error; check" + filename)
        f.close()
csv05.close()

#eps=0.1
csv1 = open("moreresults0.1.csv", "w")
for i in range(100,500,1):
    filename = './logs/fashion1l32n/onelayer32n0.1-' + str(i) + '.txt'

    with open(filename) as f:
        data = f.read()
        if '  robust [=========================================================] 9/9 queries' in data:
            csv1.write(str(i) + ",1\n")
        elif 'counterexample found' in data:
            csv1.write(str(i) + ",0\n")
        else:
            print("Major error; check" + filename)
        f.close()
csv1.close()

#eps=0.5
csv5 = open("moreresults0.5.csv", "w")
for i in range(100,500,1):
    filename = './logs/fashion1l32n/onelayer32n0.5-' + str(i) + '.txt'

    with open(filename) as f:
        data = f.read()
        if '  robust [=========================================================] 9/9 queries' in data:
            csv5.write(str(i) + ",1\n")
        elif 'counterexample found' in data:
            csv5.write(str(i) + ",0\n")
        else:
            print("Major error; check" + filename)
        f.close()
csv5.close()
