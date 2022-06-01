import torch
from train_test_models import BuckeyeDataset, DataLoader

INPUT_SIZE = 1          # number of segments
OUTPUT_SIZE = 1         # total word duration
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
MOMENTUM = 0.001
MAX_LOSS = 1.
N_EPOCHS = 50000
BATCH_SIZE = 64
SHUFFLE = True
EVERY = 1000
VECS_PATH = './data/buckeye.vecs'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = BuckeyeDataset("./data/train.jsonl", VECS_PATH)
test_data = BuckeyeDataset("./data/test.jsonl", VECS_PATH)
train_dataloader = DataLoader(
    training_data, batch_size=64, shuffle=SHUFFLE, collate_fn=lambda x: x
)

class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
        
####################################################
### MODEL WITH JUST LENGTH IN NUMBER OF SEGMENTS ###
model = LinearModel(INPUT_SIZE, OUTPUT_SIZE).to(device)

for i in range(N_EPOCHS):
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # make sure we "zero out" the loss at each time step
    optimizer.zero_grad()
    batch_data = next(iter(train_dataloader))
    xs, ys = [], []
    for segments, embedding, duration in batch_data:
        xs.append(len(segments))
        ys.append(duration)
    xs = torch.Tensor(xs).reshape(-1, 1).to(device)
    ys = torch.Tensor(ys).reshape(-1, 1).to(device)
    loss = criterion(model(xs), ys)
    if i % EVERY==0:
        print(f"Loss: {loss}")
    # do backprop over that loss
    loss.backward()
    b, m = model.parameters()
    if i % EVERY==0:
        print(f"Intercept: {b.detach()[0]}, Slope: {m.detach()[0]}")
    # move on to the next time step
    optimizer.step()

## TEST ##
test_criterion = torch.nn.MSELoss()
test_xs, test_ys = [], []
for segments, embedding, duration in test_data:
    test_xs.append(len(segments))
    test_ys.append(duration)
test_xs = torch.Tensor(test_xs).reshape(-1, 1).to(device)
test_ys = torch.Tensor(test_ys).reshape(-1, 1).to(device)
test_loss = test_criterion(model(test_xs), test_ys).detach()
print(f"Final loss on the test data is: {loss}") # tensor(0.6382)

#####################################################################
### MODEL WITH LENGTH IN NUMBER OF SEGMENTS PLUS WORD EMBEDDINGS ****
EMBEDDING_SIZE = 50
model = LinearModel(INPUT_SIZE + EMBEDDING_SIZE, OUTPUT_SIZE)

for i in range(N_EPOCHS):
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # make sure we "zero out" the loss at each time step
    optimizer.zero_grad()
    batch_data = next(iter(train_dataloader))
    xs, ys = [], []
    for segments, embedding, duration in batch_data:
        n_segments = len(segments)
        # combine two input spaces
        big_x = torch.concat([torch.Tensor([n_segments]), torch.Tensor(embedding)])
        xs.append(big_x)
        ys.append(duration)
    xs = torch.stack(xs).to(device)
    ys = torch.Tensor(ys).reshape(-1, 1).to(device)
    loss = criterion(model(xs), ys)
    if i % EVERY==0:
        print(f"Loss: {loss}")
    # do backprop over that loss
    loss.backward()
    b, m = model.parameters()
    if i % EVERY==0:
        print(f"Intercept: {b.detach()[0]}, Slope: {m.detach()[0]}")
    # move on to the next time step
    optimizer.step()

## TEST ##
test_criterion = torch.nn.MSELoss()
test_xs, test_ys = [], []
for segments, embedding, duration in batch_data:
    n_segments = len(segments)
    # combine two input spaces
    big_x = torch.concat([torch.Tensor([n_segments]), torch.Tensor(embedding)])
    test_xs.append(big_x)
    test_ys.append(duration)
test_xs = torch.stack(test_xs).to(device)
test_ys = torch.Tensor(test_ys).reshape(-1, 1).to(device)
test_loss = test_criterion(model(test_xs), test_ys).detach()
print(f"Final loss on the test data is: {loss}") # 0.6157650947570801