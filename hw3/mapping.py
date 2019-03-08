#! /usr/bin/python

import sys

# print(sys.argv[1])
# print(sys.argv[2])

dictionary = {}
with open(sys.argv[1], 'r', encoding = 'BIG5-HKSCS') as f:
	for line in f:
		sep = line.split(' ')
		ch_c = sep[0]
		zylist = sep[1].split('/')
		zylist[-1] = zylist[-1][:-1] #remove '\n'
		for subzy in zylist:
			if subzy[0] in dictionary:
				dictionary[subzy[0]].append(ch_c)
			else:
				dictionary[subzy[0]] = [ch_c]
			dictionary[ch_c] = [ch_c]

with open(sys.argv[2], 'w', encoding = 'BIG5-HKSCS') as out:
	for key in dictionary.keys():
		print(key, ' ', ' '.join(dictionary[key]), file = out)