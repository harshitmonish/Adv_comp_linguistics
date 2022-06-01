import numpy as np
import torch
from collections import Counter
from train_test_models import BuckeyeDataset, DataLoader
import pickle
import dill
from word2vec import Vocab
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

OUTPUT_SIZE = 1         # total word duration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
MOMENTUM = 0.001
MAX_LOSS = 1.
MAX_NORM = 1.
N_EPOCHS = 4
BATCH_SIZE = 64
SHUFFLE = True
DEVICE = 'cpu'
EMBEDDING_SIZE = 8
MAX_NORM = 1
EVERY = 1000
#EMBEDDING_FILE = './data/phone_weights.pt'
EMBEDDING_FILE = '../../Lab2/word2_Vec_Lab_model.pt'
#PHONE_VOCAB_FILE = './data/phones.vocab'
PHONE_VOCAB_FILE = './lab_2_data/word2_Vec_Lab_model.vocab'
TRAIN_PATH = './data/train.jsonl'
TEST_PATH = './data/test.jsonl'
VECS_PATH = "./data/buckeye.vecs"

# Fetching the data
training_data = BuckeyeDataset(TRAIN_PATH, VECS_PATH)
test_data = BuckeyeDataset(TEST_PATH, VECS_PATH)
train_dataloader = DataLoader(
    training_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=lambda x: x
)
test_dataloader = DataLoader(
    test_data, batch_size=BATCH_SIZE, collate_fn= lambda x:x
)
#phone_embeds = torch.load(EMBEDDING_FILE)


#phone_embeds, phone_vocab = train_model()
#embedding_dim = phone_embeds.embedding.weight.size()[-1]


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

class Vocab():
    def __init__(self, segments: list):
        self._compute_frequency_table(segments)
        print(self.frequency_table)
        self._build_ix_to_vocab_dicts()

    def _compute_frequency_table(self, segments):
        self.frequency_table = Counter(segments)
        self.vocab_size = len(self.frequency_table)
    
    def _build_ix_to_vocab_dicts(self):
        self.ix_to_vocab = {
            i: phone for i, phone in enumerate(self.frequency_table)
            if self.frequency_table[phone] > 0
        }
        self.vocab_to_ix = {
            self.ix_to_vocab[w]: w for w in self.ix_to_vocab.keys()
        }

    def tokenize(self, list_of_segments):
        return torch.tensor(
            [self.vocab_to_ix[w] for w in list_of_segments], dtype=torch.long,
            device=DEVICE).view(-1, 1)
    
    def detokenize(self, tensor):
        return torch.tensor(
            [self.ix_to_vocab[ix] for ix in tensor], dtype=torch.long,
            device=DEVICE).view(-1, 1)


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


class MultilayerDurationModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int=None, max_norm=None):
        super(MultilayerDurationModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 64) # flat
        self.hidden = torch.nn.Linear(64, output_size)
        # TODO: Add an intermediate layer

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear(x))
        x = self.hidden(x)
        # TODO: Add the processing of the intermediate layer
        return x


def tokenize(vocab, list_of_segments):
    return torch.tensor(
        [vocab.vocab_to_ix[w] for w in list_of_segments if w in vocab.frequency_table], dtype=torch.long,
        device=DEVICE).view(-1, 1)

# Fetching the vocab and word2Vec model
phone_vocab = dill.load(open(PHONE_VOCAB_FILE, 'rb'))
phone_embeds = Word2Vec(
        input_size=phone_vocab.vocab_size,
        embedding_size=EMBEDDING_SIZE,
        max_norm=MAX_NORM
        )
phone_embeds.load_state_dict(torch.load(EMBEDDING_FILE))
embedding_dim = phone_embeds.embedding.weight.size()[-1]


####################################################

print("Training The Model ")
max_training_length = max([len(s[0]) for s in training_data]) # TODO: Figure out this number from the training data

#####################################################################
### LINEAR MODEL EMBEDDINGS CONCATENATED FOR EACH SEGMENT        ****
#####################################################################

modelL = LinearModel(max_training_length * embedding_dim, OUTPUT_SIZE).to(DEVICE)

#model = LinearModel(embedding_dim, OUTPUT_SIZE)  # TODO: Figure out the right shape
embedding_matrix = phone_embeds.embedding.weight.detach()
input_dims = max_training_length * embedding_dim

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(modelL.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

for i in range(N_EPOCHS):
    print("EPOCH : " + str(i))
    running_loss = 0.0
    for idx, batch_data in enumerate(train_dataloader, 0):
        xs, ys = [], []

        # make sure we "zero out" the loss at each time step
        optimizer.zero_grad()
        batch_segments = []
        for segments, embedding, duration in batch_data:
            # TODO: concatenate the phone embeddings associated with all segments
            # TIP: reference your pt & vocab files
            segments_embed = []
            if len(segments) > 0:
                #concatenating all the segment embedding into a single tensor
                for segment in segments:
                    segment_id = tokenize(phone_vocab, segment)
                    segment_embedding = embedding_matrix[segment_id].flatten().tolist()
                    segments_embed.append(segment_embedding)
            #embeds = torch.Tensor(np.array(np.concatenate(segments_embed).flat)).flatten()
            embeds = list(np.concatenate(segments_embed).flat)
            batch_segments.append(embeds)

            #Adding padding to make it of the size of largest sequence tensor
            #padding_len = int((input_dims - (embeds.shape[0]))/2)
            #padding = torch.nn.ConstantPad1d(padding_len, 0.)

            #ConstPad1d() takes care of tensors that are bigger than max size.
            #padded_embeds = padding(embeds)
            ys.append(duration)

        # Adding padding to make it of the size of largest sequence tensor
        padded_embeds = torch.Tensor(pad_sequences(batch_segments, value=0., padding='post', truncating='post', maxlen=input_dims, dtype='float32'))
        #xs = torch.stack(xs)
        ys = torch.Tensor(ys).reshape(-1, 1).to(DEVICE)
        loss = criterion(modelL(padded_embeds), ys)
        running_loss += loss.item()
        if idx % EVERY==0:
            print(f"Loss: {loss}")

        # do backprop over that loss
        loss.backward()
        #b, m = model.parameters()
        #if i % EVERY==0:
        #    print(f"Intercept: {b.detach()[0]}, Slope: {m.detach()[0]}")
        # move on to the next time step
        optimizer.step()
    print(loss.item())

print("\n DONE Training ")

## TEST ##
print("\n \n Evaluating the Linear model on Test Data")
modelL.eval()

for idx, batch_data in enumerate(test_dataloader, 0):
    xs, ys = [], []
    optimizer.zero_grad()
    batch_segments = []
    for segments, embedding, duration in batch_data:
        # TODO: concatenate the phone embeddings associated with all segments
        # TIP: reference your pt & vocab files
        segments_embed = []
        if len(segments) > 0:
            # concatenating all the segment embedding into a single tensor
            for segment in segments:
                segment_id = tokenize(phone_vocab, segment)
                segment_embedding = embedding_matrix[segment_id].flatten().tolist()
                segments_embed.append(segment_embedding)

        embeds = list(np.concatenate(segments_embed).flat)
        batch_segments.append(embeds)

        # Adding padding to make it of the size of largest sequence tensor
        #padding_len = int((input_dims - (embeds.shape[0])) / 2)
        #padding = torch.nn.ConstantPad1d(padding_len, 0.)
        ys.append(duration)

    padded_embeds = torch.Tensor(pad_sequences(batch_segments, value=0., padding='post', truncating='post', maxlen=input_dims, dtype='float32'))
    ys = torch.Tensor(ys).reshape(-1, 1).to(DEVICE)
    loss = criterion(modelL(padded_embeds), ys)
    if idx % EVERY == 0:
        print(f"Loss: {loss}")

print(f"Final loss on the test data  for linear model is: {loss}")
torch.save(modelL.state_dict(), "./data/cbow_linear_model.pt")

#####################################################################
### MULTILAYER MODEL EMBEDDINGS CONCATENATED FOR EACH SEGMENT    ****
#####################################################################

modelM = MultilayerDurationModel(max_training_length * embedding_dim, OUTPUT_SIZE).to(DEVICE)
#model = LinearModel(embedding_dim, OUTPUT_SIZE)  # TODO: Figure out the right shape
embedding_matrix = phone_embeds.embedding.weight.detach()
input_dims = max_training_length * embedding_dim

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(modelM.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

for i in range(N_EPOCHS):
    print("EPOCH : " + str(i))
    running_loss = 0.0
    for idx, batch_data in enumerate(train_dataloader, 0):
        xs, ys = [], []

        # make sure we "zero out" the loss at each time step
        optimizer.zero_grad()
        batch_segments = []
        for segments, embedding, duration in batch_data:
            # TODO: concatenate the phone embeddings associated with all segments
            # TIP: reference your pt & vocab files
            segments_embed = []
            if len(segments) > 0:
                #concatenating all the segment embedding into a single tensor
                for segment in segments:
                    segment_id = tokenize(phone_vocab, segment)
                    segment_embedding = embedding_matrix[segment_id].flatten().tolist()
                    segments_embed.append(segment_embedding)
            #embeds = torch.Tensor(np.array(np.concatenate(segments_embed).flat)).flatten()
            embeds = list(np.concatenate(segments_embed).flat)
            batch_segments.append(embeds)

            #Adding padding to make it of the size of largest sequence tensor
            #padding_len = int((input_dims - (embeds.shape[0]))/2)
            #padding = torch.nn.ConstantPad1d(padding_len, 0.)

            #ConstPad1d() takes care of tensors that are bigger than max size.
            #padded_embeds = padding(embeds)
            ys.append(duration)

        # Adding padding to make it of the size of largest sequence tensor
        padded_embeds = torch.Tensor(pad_sequences(batch_segments, value=0., padding='post', truncating='post', maxlen=input_dims, dtype='float32'))
        #xs = torch.stack(xs)
        ys = torch.Tensor(ys).reshape(-1, 1).to(DEVICE)
        loss = criterion(modelM(padded_embeds), ys)
        running_loss += loss.item()
        if idx % EVERY==0:
            print(f"Loss: {loss}")

        # do backprop over that loss
        loss.backward()
        #b, m = model.parameters()
        #if i % EVERY==0:
        #    print(f"Intercept: {b.detach()[0]}, Slope: {m.detach()[0]}")
        # move on to the next time step
        optimizer.step()
    print(loss.item())

print("\n DONE Training ")

## TEST ##
print("\n \n Evaluating the Multi layer model on Test Data")
modelM.eval()

for idx, batch_data in enumerate(test_dataloader, 0):
    xs, ys = [], []
    optimizer.zero_grad()
    batch_segments = []
    for segments, embedding, duration in batch_data:
        # TODO: concatenate the phone embeddings associated with all segments
        # TIP: reference your pt & vocab files
        segments_embed = []
        if len(segments) > 0:
            # concatenating all the segment embedding into a single tensor
            for segment in segments:
                segment_id = tokenize(phone_vocab, segment)
                segment_embedding = embedding_matrix[segment_id].flatten().tolist()
                segments_embed.append(segment_embedding)

        embeds = list(np.concatenate(segments_embed).flat)
        batch_segments.append(embeds)

        # Adding padding to make it of the size of largest sequence tensor
        #padding_len = int((input_dims - (embeds.shape[0])) / 2)
        #padding = torch.nn.ConstantPad1d(padding_len, 0.)
        ys.append(duration)

    padded_embeds = torch.Tensor(pad_sequences(batch_segments, value=0., padding='post', truncating='post', maxlen=input_dims, dtype='float32'))
    ys = torch.Tensor(ys).reshape(-1, 1).to(DEVICE)
    loss = criterion(modelM(padded_embeds), ys)
    if idx % EVERY == 0:
        print(f"Loss: {loss}")
print(f"Final loss on the test data for multilayer model is: {loss}")

torch.save(modelM.state_dict(), "./data/cbow_multi_model.pt")

"""
test_criterion = torch.nn.MSELoss()
test_xs, test_ys = [], []
for segments, embedding, duration in batch_data:
    # TODO: concatenate the phone embeddings associated with all segments
    # TIP: reference your pt & vocab files
    if len(segments) > 0:
        first_segment_id = tokenize(phone_vocab, segments)[0]  # TODO: Make this match training
        first_segment_embedding = embedding_matrix[first_segment_id] # TODO: Make this match training
        test_xs.append(first_segment_embedding)
        test_ys.append(duration)

test_xs = torch.stack(test_xs)[:, 0, :]
test_ys = torch.Tensor(test_ys).reshape(-1, 1)
test_loss = test_criterion(model(test_xs), test_ys).detach()
print(f"Final loss on the test data is: {loss}") # tensor(0.6382)"""