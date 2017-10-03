f = open('words.txt', 'r')
c = f.readlines()
terms = c[0].split()

wordDict = []
wordCout = []
for i in range(len(terms)):
	chk = terms[i]
	if (chk not in wordDict):
		wordDict.append(chk)
		wordCout.append(1)
	else:
		wordCout[wordDict.index(chk)] += 1

assert len(wordDict) == len(wordCout)

with open('Q1.txt', 'w') as fout:
	for i in range(len(wordDict)):
		line = wordDict[i]+" "+str(i)+" "+str(wordCout[i])+"\n"
		fout.write(line)

f.close()
fout.close()