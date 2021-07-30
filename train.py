# imports
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from model import Model
import torch


def train(model, dataset, optimizer, criterion, loader):
    model.train()
    for epoch in range(2):
        for batch in loader:
            optimizer.zero_grad()
            loss = 0
            h = model.init_states(batch.shape[0])
            for i in range(batch.shape[1] - 1):
                # source (reshape for samples on different rows)
                x = torch.reshape(batch[:, i], (batch.shape[0], 1))
        
                # target (reshape for samples on different rows)
                y = torch.reshape(batch[:, i + 1], (batch.shape[0], 1))

                #  
                p, h = model(x, h)


                p = p[:, 0, :]

                loss += criterion(p, y.squeeze(1))
            loss.backward()
        
            print(loss)

        optimizer.step()



# preict words (based on having trained on like one book)
def predict(dataset, model, text):

    # put model in evaluation mode 
    model.eval()
    
    # text
    words = text.split(' ') 

    # text to index
    prompt = torch.tensor([dataset.word_to_index[word] for word in words])
    
    # initiate hidden states
    h = model.init_states(1)

    for i in range(len(prompt)):
        x = prompt[i][None, None]
        p, h = model(x, h)


    s = F.softmax(p, dim=2)[0, 0, :]
    word = dataset.index_to_word[torch.argmax(s).item()]
    print(word)
    
    
def main():
    dataset = Dataset()
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    model = Model(dataset) 
    optimizer = optim.Adam(model.parameters(), lr=.001)
    criterion = nn.CrossEntropyLoss()
    # train(model, dataset, optimizer, criterion, loader)
    # torch.save(model.state_dict(), "params")

    model.load_state_dict(torch.load("params"))
    
    predict(dataset, model, "I have to go now")


if __name__ == "__main__":
    main()
