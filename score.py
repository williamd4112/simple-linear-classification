import csv, sys

'''
    Given two csv file, count classfication accuracy
    Each row in csv is one-hot vector (one-of-k coding)
'''

a_file = csv.reader(open(sys.argv[1], 'r'), delimiter=',')
b_file = csv.reader(open(sys.argv[2], 'r'), delimiter=',')

count = 0
correct = 0
for a, b in zip(a_file, b_file):
    k_a = a.index('1')
    k_b = b.index('1')
    count += 1
    if k_a == k_b:
        correct += 1
print 'Accuracy: ', (float(correct) / count) * 100.0, '%'
