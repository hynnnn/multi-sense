import io
import time, datetime
from nltk import word_tokenize
from nltk import sent_tokenize

def split(line,delta):
	size = 2 * delta + 1
	sent = sent_tokenize(line.strip().lower()) 
	for sen in sent:
		tokens = word_tokenize(sen)
		filtered_tokens = [t for t in tokens if sum([c.isalnum() for c in t])]
		length = len(filtered_tokens)
		if len(filtered_tokens) >= size:
		 	for i in range(length-size+1):
		 		new = filtered_tokens[i:i + size]
		 		fw.write(" ".join(new) + "\n")

f = io.open('input.txt', encoding='utf-8', errors='ignore')
i = 0
data = []
with io.open('./train.txt', 'w', encoding='utf-8') as fw:
	for line in f:
		i += 1
		delta = 5
		split(line, delta)