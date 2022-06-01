import pickle

import pandas as pd
import torch, json
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from torch.utils.data import DataLoader
import dill
torch.manual_seed(1988)

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
MOMENTUM = 0.00001
MAX_NORM = 1
N_EPOCHS = 1000
EMBEDDING_SIZE = 8 # TODO: Parametrically manipulate size
WINDOW_SIZE = 4
BATCH_SIZE = 250
DEVICE="cpu"

# load in the buckeye data
def process_segments_for_CBOW(line_data):
    xs, ys, flat = [], [], []
    segments = line_data['observed_pron'].split(" ")
    if len(segments) > 1:
        for i, segment in enumerate(segments):
            # define context using indices before/after
            # TODO: prev_segments: list - should be the window size prior to i
            #prev_segments = None
            prev_segments = segments[max(0, i - WINDOW_SIZE):i]
            # TODO: next_segments: list - should be the window size starting after i
            #next_segments = None
            next_segments = segments[i+1 : i+WINDOW_SIZE+1]
            target: str = segment
            context = prev_segments + next_segments
            xs.append(context)
            ys.append([target])
            # TODO: Reflect on why we use += here
            flat.append(target)
            flat += context
    return xs, ys, flat


class Vocab():
    def __init__(self, segments: list):
        self._compute_frequency_table(segments)
        self._build_ix_to_vocab_dicts()

    def _compute_frequency_table(self, segments):
        self.frequency_table = Counter(segments)
        self.vocab_size = len(self.frequency_table)
    
    def _build_ix_to_vocab_dicts(self):
        # TODO: create a dictionary that maps from words to indices using self.frequency_table
        self.ix_to_vocab = {i : ph for i, ph in enumerate(self.frequency_table)
                            if self.frequency_table[ph] > 0
                            }
        # hint: try a list comprehension or a for loop
        # TODO: create a dictionary that maps from indices to words using self.frequency_table
        self.vocab_to_ix = { ph : i for i, ph in self.ix_to_vocab.items()
                             if self.frequency_table[ph] > 0
                            }
        # hint: try a list comprehension or a for loop


    def tokenize(self, list_of_segments):
        return torch.tensor(
            [self.vocab_to_ix[w] for w in list_of_segments], dtype=torch.long)
    
    def detokenize(self, tensor):
        return torch.tensor(
            [self.ix_to_vocab[ix] for ix in tensor], dtype=torch.long)


class Word2Vec(torch.nn.Module):
    def __init__(self, input_size: int, embedding_size: int, output_size: int=None, max_norm=None):
        super(Word2Vec, self).__init__()
        self.embedding = torch.nn.Embedding(
            input_size,
            embedding_size,
            max_norm=MAX_NORM
            )
        if output_size is None:
            self.linear = torch.nn.Linear(embedding_size, input_size)
        else:
            self.linear = torch.nn.Linear(embedding_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


def main():
    buckeye = [json.loads(x) for x in open("./lab_2_data/buckeye.jsonl", "r").readlines()]
    # TODO: Change the one_monologue variable to combine all monologues from buckeye
    monologues = []

    #Taking n max monologues from the dataset.
    max_items = 80
    for i in buckeye:
        if(max_items == 0):
            break
        for _, j in i.items():
            monologues.append(j)
        max_items-=1
    contexts, targets, vocabulary = [], [], []
    # make sure this works for the aggregate representation you created
    for m in monologues:
        for _, line_data in m.items():
            xs_, ys_, vocab = process_segments_for_CBOW(line_data)
            contexts += xs_
            targets += ys_
            vocabulary += vocab

    vocab = Vocab(vocabulary)
    model = Word2Vec(
        input_size=vocab.vocab_size,
        embedding_size=EMBEDDING_SIZE,
        max_norm=MAX_NORM
        )
    x_tensors = pad_sequence([vocab.tokenize(x) for x in contexts]).t()
    y_tensors = pad_sequence([vocab.tokenize(x) for x in targets]).t().reshape(-1)

    y_tensors = y_tensors.reshape((y_tensors.shape[0],1))
    print("X, Y shape: ")
    print(x_tensors.shape, y_tensors.shape)
    train_data = torch.cat((x_tensors, y_tensors), 1)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train_dl = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    for i in range(N_EPOCHS):
        # TODO: implement random sampling using torch.randperm
        # Used Dataloader
        # TODO: manipulate batch size
        for batch_idx, batch in enumerate(train_dl):
            x = batch[:,:-1].to(DEVICE)
            y = batch[:,-1].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {i} Loss: {loss}")



    # TODO: use torch.save to store your trained model
    #torch.save(model, "./word2_Vec_Lab_model.pt")
    torch.save(model.state_dict(), "./lab_2_data/word2_Vec_Lab_model.pt")

    # TODO: use pickle to save the vocabulary object
    with open('./lab_2_data/word2_Vec_Lab_model.vocab', 'wb') as file_obj:
        dill.dump(vocab, file_obj)

if __name__ == "__main__":
    main()
