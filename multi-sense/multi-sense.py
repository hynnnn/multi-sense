import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import sys
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity


def find_split(list):
	local_min = []
	for i in range(1,len(list)):
		if list[i]!=list[i-1]+1:
			local_min.append(list[i]-1)
			list.append(list[i]-1)
	local_min.append(min(list)-1)				
	list.append(min(list)-1)
	list.sort()
	return list, local_min

def combine_embedding(value, local_min, lnew):
	output = []
	count = 0
	c= torch.zeros(dim, dtype=torch.float)
	for i, k in enumerate(value):
		if i not in lnew:
			if count != 0 and sum(c)!=0:
				output.append(c)
				c = torch.zeros(dim, dtype=torch.float)
			output.append(k)
		if i in lnew:
			if i in local_min:
				if sum(c)!=0:
					output.append(c)
				count += 1
				c = torch.zeros(dim, dtype=torch.float)
			c += k
	if sum(c)!= 0:
		output.append(c)
	return output

if __name__ == "__main__":

	file_name = sys.argv[1]
	data = open(file_name,"r")

	dataset = []
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	dim = 768

	for line in  data:
		dataset.append(line.strip())

	for sentence in dataset:
		sen = sentence.split(" ")
		tokenized_text = tokenizer.tokenize(sentence)
		print(sentence)
		print(tokenized_text)

		#find the split index
		split_id = []
		for i,x in enumerate(tokenized_text):
			if x[0] == "#":
				split_id.append(i)
		if len(split_id) != 0 :
			split_id, local_min = find_split(split_id)
			print ("The split index is: ", split_id)

		indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

		length = len(tokenized_text)
		segments_ids = [1] * length

		tokens_tensor = torch.tensor([indexed_tokens])
		segments_tensors = torch.tensor([segments_ids])

		# Load pre-trained model (weights)
		model = BertModel.from_pretrained('bert-base-uncased')
		model.eval()

		with torch.no_grad():
			encoded_layers, _ = model(tokens_tensor, segments_tensors)

		# Convert the hidden state embeddings into single token vectors
		# Holds the list of 12 layer embeddings for each token
		# Will have the shape: [# tokens, # layers, # features]
		token_embeddings = [] 
		# For each token in the sentence...
		for token_i in range(length):
			# Holds 12 layers of hidden states for each token 
			hidden_layers = [] 
			# For each of the 12 layers...
			for layer_i in range(len(encoded_layers)):
			# Lookup the vector for `token_i` in `layer_i`
				vec = encoded_layers[layer_i][0][token_i]
				hidden_layers.append(vec)
			token_embeddings.append(hidden_layers)

		# Sanity check the dimensions:
		print ("Number of tokens in sequence:", len(token_embeddings))
		print ("Number of layers per token:", len(token_embeddings[0]))
		print ("dimensions per token:", len(token_embeddings[0][0]))

		# Stores the token vectors, with shape [22 x 768]
		token_vecs_sum = []
		# For each token in the sentence...
		for token in token_embeddings:
		    # Sum the vectors from the last four layers.
		    sum_vec = torch.sum(torch.stack(token)[-4:], 0)		    
		    # Use `sum_vec` to represent `token`.
		    token_vecs_sum.append(sum_vec)
		print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


		combine = torch.empty(dim, dtype=torch.float)
		if len(split_id)!=0:
			final_embedding = combine_embedding(token_vecs_sum, local_min, split_id)
		print("Number of token after combining:",len(final_embedding))

		#final_embedding is the final context-based embedding that extracted from BERT
		length = len(final_embedding)
		ind = int((length-1)/2)
		#embedding for the center word
		v_bert = final_embedding[ind]
		print("The center word has the dimensions:", v_bert.shape)

		#get the indexed_tokens for the center word
		center_word = sen[ind]
		print("The center word is '%s'"%format(center_word))
		indexed_tokens = tokenizer.convert_tokens_to_ids([center_word])
		print("The index of the center word:", indexed_tokens)

		context_embedding = final_embedding[0]
		for i in range(1, length):
			if i != ind:
				context_embedding = torch.cat((context_embedding,final_embedding[i]),0)
		print("After concatnate the context, the vector has dimension:",context_embedding.shape)

		#to be clear, I loop for a single input first
		break
