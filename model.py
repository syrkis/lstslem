# model.py
#   neural language model fun
# by: Noah Syrkis

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    # dimensions and stuff
    embedding_dim = 2 ** 5
    hidden_size = 2 ** 5
    num_layers = 2
    
    def __init__(self, dataset):
        super().__init__() 

        # embedding layer
        self.embed =  nn.Embedding(
            len(dataset.vocab),
            self.embedding_dim,
          #  padding_idx=dataset.word_to_index['<PAD>'],
        )

        # lstm layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        ) 

        # linear layer
        self.fc = nn.Linear(
            self.hidden_size,
            len(dataset.vocab),
        )
    
    def forward(self, x, h):

        # embed input
        x = self.embed(x)

        # lstm (h is hidden state)
        x, h = self.lstm(x) 

        # linear layer
        x = self.fc(x)

        # return output
        return x, h
  
    def init_states(self, batch_size):
        # initial hidden state for lstm
        return (torch.randn(self.num_layers, batch_size, self.hidden_size),
                torch.randn(self.num_layers, batch_size, self.hidden_size))


def main():
    from dataset import Dataset
    from tqdm import tqdm

    batch_size = 2 ** 4
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = Model(dataset)
    h = model.init_states(batch_size)
    for batch in loader:
        for i in range(batch.shape[1] - 1): 
            x = torch.reshape(batch[:, i], (batch.shape[0], 1))
            y = torch.reshape(batch[:, i + 1], (batch.shape[0], 1))
            p, h = model(x, h)
            print(p.shape)
            break
        break
        
if __name__ == '__main__':
    main()
