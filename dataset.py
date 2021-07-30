# dataset.py
#   pytorch dataset fun
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize, sent_tokenize


class Dataset(torch.utils.data.Dataset):

    # sentence length 
    sent_len = 2 ** 5
    
    def __init__(self):  
        # open data
        with open('nostromo.txt', 'r') as f:
            # read data
            book = f.read() 
        # split data into sentences
        sents = sent_tokenize(book)  
        # split sentences into tokens
        self.tokens = [word_tokenize(sent) for sent in sents]
        # setup vocabulary
        self.vocab = {'<PAD>', '<UNK>', '<S>', '</S>', '<EOS>'}
        # loop through sentences
        for sent in self.tokens:
            # loop through words in each sentence
            for word in sent:
                # add word to vocab
                self.vocab.add(word)
        # word to index dict
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        # index to word dict
        self.index_to_word = {idx: word for idx, word in enumerate(self.vocab)} 
         
    def __len__(self):
        # return sample (setence) count of data set
        return len(self.tokens)

    def __getitem__(self, idx):
        # get sentenc eof index idx and convert word to word indexes 
        sent = [self.word_to_index[word] for word in self.tokens[idx]]  
        # truncate sentences that are too long
        if len(sent) > self.sent_len:
            sent = sent[:self.sent_len]
        # pad sentences that are too short
        elif len(sent) < self.sent_len:
            tmp = [self.word_to_index["<PAD>"]] * (self.sent_len - len(sent))
            tmp.extend(sent)
            sent = tmp
        # return torch tensor version of sentence of word indexes
        return torch.tensor(sent) 


def main():
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch)
        

if __name__ == '__main__':
    main()
