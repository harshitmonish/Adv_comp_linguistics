import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import pickle, time, math, random, json
from train_test_models import BuckeyeDataset, DataLoader
import dill
import torch.nn.functional as F

SOS_TOKEN = 0
EOS_TOKEN = 1
OUTPUT_SIZE = 1  # total word duration
MAX_LENGTH = 18
HIDDEN_SIZE = 8
DEVICE = 'cpu'
MAX_NORM = 1

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0
MOMENTUM = 0.0001
MAX_LOSS = 1.
N_EPOCHS = 500
BATCH_SIZE = 64
SHUFFLE = True
EMBEDDING_SIZE = 8
EVERY = 50
GLOVE_EMBED_DIM = 50

EMBEDDING_FILE = './lab_2_data/word2_Vec_Lab_model.pt'
PHONE_VOCAB_FILE = './lab_2_data/word2_Vec_Lab_model.vocab'
TRAIN_PATH = './data/train.jsonl'
TEST_PATH = './data/test.jsonl'
VECS_PATH = "./data/buckeye.vecs"
SHUFFLE_DATA = True


def tokenize(vocab, list_of_segments):
    return torch.tensor(
        [vocab.vocab_to_ix[w] for w in list_of_segments if w in vocab.frequency_table] + [EOS_TOKEN], dtype=torch.long,
        device=DEVICE).view(-1, 1)


def process_segments_for_encoder(line_data):
    # Split every line into pairs and normalize
    split_str = line_data['observed_pron'].split(" ")
    duration = torch.log(torch.Tensor([sum(line_data['segment_duration_ms'])]))
    return (split_str, duration)


def tensorFromSentence(vocab, sentence):
    indices = tokenize(vocab, sentence)
    return indices


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
    def __init__(self, input_size: int, embedding_size: int, output_size: int = None, max_norm=None):
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


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class EncoderGloveRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, glove_size=None):
        super(EncoderGloveRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        if glove_size is not None:
            self.linear = torch.nn.Linear(glove_size, hidden_size)
            self.gru = torch.nn.GRU(hidden_size * 2, hidden_size)
        else:
            self.linear = torch.nn.Linear(0, hidden_size)
            self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, glove_embedding=None):
        embedded = self.embedding(input).view(1, 1, -1)
        if glove_embedding is not None:
            linear = self.linear(glove_embedding).view(1, 1, -1)
            # print(embedded.shape, linear.shape)
            output = torch.cat((embedded, linear), axis=2)
        else:
            output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class EncoderRNN3(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embeddings=None):
        super(EncoderRNN3, self).__init__()
        self.hidden_size = hidden_size
        if embeddings is not None:
            input_size, hidden_size = embeddings.size()
            self.embedding = torch.nn.Embedding(input_size, hidden_size)
            self.embedding.weight = torch.nn.Parameter(embeddings)
        else:
            self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=DEVICE)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, glove_embedding=None):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max(input_length, max_length), encoder.hidden_size, device=DEVICE)

    loss = 0

    for ei in range(input_length):
        if (type(encoder) is EncoderGloveRNN):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden, glove_embedding)
        else:
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=DEVICE)

    decoder_hidden = encoder_hidden

    # use its own predictions as the next input
    for di in range(target_length):
        if type(decoder) is DecoderRNN:
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        elif type(decoder) is AttnDecoderRNN:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di].view(1))
        if decoder_input.item() == EOS_TOKEN:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return encoder, decoder, loss.item() / target_length


def trainIters(pairs, encoder, decoder, learning_rate=0.01):
    start = time.time()

    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = torch.nn.NLLLoss()

    for training_pair in pairs:
        input_tensor, target_tensor, glove_embedding = training_pair

        encoder, decoder, loss = train(
            input_tensor, target_tensor, encoder,
            decoder, encoder_optimizer, decoder_optimizer, criterion, glove_embedding=torch.Tensor(glove_embedding))


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    # fetching the data
    training_data = BuckeyeDataset(TRAIN_PATH, VECS_PATH)
    test_data = BuckeyeDataset(TEST_PATH, VECS_PATH)
    train_dataloader = DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA, collate_fn=lambda x: x
    )
    # loading the Vocab class
    phone_vocab = dill.load(open(PHONE_VOCAB_FILE, 'rb'))

    # loading learned phone embeddings from CBOW
    phone_embeds = Word2Vec(
        input_size=phone_vocab.vocab_size,
        embedding_size=EMBEDDING_SIZE,
        max_norm=MAX_NORM
    )
    phone_embeds.load_state_dict(torch.load(EMBEDDING_FILE))
    embedding_dim = phone_embeds.embedding.weight.size()[-1]

    # phone_embeds = torch.load(EMBEDDING_FILE)

    # phone_vocab = pickle.load(open(PHONE_VOCAB_FILE, 'rb'))
    # embedding_dim = phone_embeds.embedding.weight.size()[-1]

    # evaluating the max length
    MAX_LENGTH = max([len(s[0]) for s in training_data])

    # Encoder-decoder hidden states
    # Encoder-decoder hidden states + learned CBOW embeddings
    # Encoder-decoder hidden states + learned phone embeddings + GloVe embeddings
    # Encoder-decoder with attention hidden states
    # Encoder-decoder with attention hidden states + GloVe embeddings
    # Encoder-decoder with attention hidden states + learned CBOW embeddings + GloVe embeddings

    n_words_for_encoders = phone_vocab.vocab_size + len([SOS_TOKEN, EOS_TOKEN])
    encoder1 = EncoderRNN(n_words_for_encoders, HIDDEN_SIZE).to(DEVICE)
    encoder2 = EncoderGloveRNN(n_words_for_encoders, HIDDEN_SIZE, GLOVE_EMBED_DIM).to(DEVICE)
    encoder3 = EncoderRNN3(n_words_for_encoders, HIDDEN_SIZE,
                           embeddings=torch.nn.Parameter(phone_embeds.embedding.weight)).to(DEVICE)

    decoder1 = DecoderRNN(HIDDEN_SIZE, n_words_for_encoders)
    attn_decoder1 = AttnDecoderRNN(HIDDEN_SIZE, n_words_for_encoders, dropout_p=0.1).to(DEVICE)
    print("Training Encoder Decoder")
    for encoder in [encoder1, encoder2, encoder3]:  # , encoder3]: # , encoder2]:
        for decoder in [decoder1]:  # , attn_decoder1]:
            for i in range(N_EPOCHS):
                batch_data = next(iter(train_dataloader))
                xs, ys, glove_embeddings = [], [], []
                for segments, glove_embedding, duration in batch_data:
                    if len(segments) > 0:
                        xs.append(tensorFromSentence(phone_vocab, segments))
                        ys.append(tensorFromSentence(phone_vocab, segments))
                        glove_embeddings.append(glove_embedding)
                # print(glove_embedding, embedding_dim)
                pairs = list(zip(xs, ys, glove_embeddings))
                trainIters(pairs, encoder, decoder)
                # evaluateRandomly(encoder, decoder, pairs, phone_vocab, n=10)
                encoder_type = str(type(encoder)).split(".")[-1][:-2]
                decoder_type = str(type(decoder)).split(".")[-1][:-2]
                # torch.save(encoder, f"./2022-05-11_{type(encoder)}_{type(decoder)}.pt")
                torch.save(encoder, f"./2022-05-11_{encoder_type}_{decoder_type}.pt")
                # torch.save(encoder.state_dict(), f"./2022-05-11_{type(encoder)}_{type(decoder)}.pt")

    # train duration prediction model
    model = LinearModel(HIDDEN_SIZE, OUTPUT_SIZE)
    embedding_matrix = phone_embeds.embedding.weight.detach()
    print("Training model")
    for ix, encoder in enumerate([encoder1, encoder2, encoder3]):  # , encoder3]): #, encoder3]):
        encoder.eval()
        encoder_hidden = encoder.initHidden()
        for i in range(N_EPOCHS):
            loss = 0
            criterion = torch.nn.MSELoss()
            # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
            # make sure we "zero out" the loss at each time step
            optimizer.zero_grad()
            batch_data = next(iter(train_dataloader))
            xs, ys = [], []
            batch_segments = []
            # for idx, batch_data in enumerate(train_dataloader, 0):
            for segments, embedding, duration in batch_data:
                input_length = len(segments)
                encoder_outputs = torch.zeros(input_length, HIDDEN_SIZE,
                                              device=DEVICE, requires_grad=False)
                if input_length > 0:
                    input_tensor = tokenize(phone_vocab, segments)
                    for ei in range(input_length):
                        if (type(encoder) is EncoderGloveRNN):
                            encoder_output, encoder_hidden = encoder(
                                input_tensor[ei], encoder_hidden, torch.Tensor(embedding))
                        else:
                            encoder_output, encoder_hidden = encoder(
                                input_tensor[ei], encoder_hidden)
                        encoder_outputs[ei] += encoder_output[0, 0]
                        last_hidden_state = encoder_outputs[-1].detach()
                        xs.append(last_hidden_state.flatten())
                        ys.append(duration)
            xs = torch.stack(xs)
            ys = torch.Tensor(ys).reshape(-1, 1)
            loss = criterion(model(xs), ys)

            if i % EVERY == 0:
                print(f"Loss: {loss}")

            loss.backward()
            b, m = model.parameters()
            optimizer.step()

        print(f"Final loss on the training data for encoder {ix} is: {loss}")  # tensor(0.478)
        ## TEST ##
        test_criterion = torch.nn.MSELoss()
        test_xs, test_ys = [], []
        for segments, embedding, duration in batch_data:
            input_length = len(segments)
            encoder_outputs = torch.zeros(input_length, HIDDEN_SIZE,
                                          device=DEVICE, requires_grad=False)
            if input_length > 0:
                input_tensor = tokenize(phone_vocab, segments)
                for ei in range(input_length):
                    if (type(encoder) is EncoderGloveRNN):
                        encoder_output, encoder_hidden = encoder(
                            input_tensor[ei], encoder_hidden, torch.Tensor(embedding))
                    else:
                        encoder_output, encoder_hidden = encoder(
                            input_tensor[ei], encoder_hidden)

                        encoder_outputs[ei] += encoder_output[0, 0]
                last_hidden_state = encoder_outputs[-1].detach()
            test_xs.append(last_hidden_state)
            test_ys.append(duration)
        test_xs = torch.stack(test_xs)
        test_ys = torch.Tensor(test_ys).reshape(-1, 1)
        test_loss = test_criterion(model(test_xs), test_ys).detach()

        print(f"Final loss on the test data for encoder {ix} is: {test_loss}")  # tensor(0.7235)

if __name__ == "__main__":
    main()

