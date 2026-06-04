import torch
import time
import torchvision
from torch import nn 
from torchvision.transforms import v2
torch.manual_seed(42)
torch.set_num_threads(11)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)])
data_train = torchvision.datasets.CIFAR10(root=".", train=True, download=True, transform=transform)
data_test = torchvision.datasets.CIFAR10(root =".", train=False, download=True, transform=transform)

def load(data):
    x = torch.stack([data[i][0].reshape(-1) for i in range(len(data))])
    y = torch.tensor([data[i][1] for i in range(len(data))], dtype=torch.long)
    
    return x, y

X_train, Y_train = load(data_train)
X_test, Y_test = load(data_test)
class NeuralNetwork(nn.Module):
    def __init__(self, hidden = 128):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3072, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden // 4, 10)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

def init_weights(layer):
    if (isinstance(layer, nn.Linear)):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

model = NeuralNetwork(hidden=128)
model.apply(init_weights)
EPOCHS = 1000
BATCH = 512

# model.to(device)
lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
Scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100*BATCH)

history = []
accuracy = []
i = 0

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    perm = torch.randperm(X_train.shape[0])
    corr = 0
    tot = 0
    
    t0 = time.perf_counter()
    for j in range(0, X_train.shape[0], BATCH):
        # get the inputs; data is a list of [inputs, labels]
        idx = perm[j:j + BATCH]
        inputs, labels = X_train[idx], Y_train[idx]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = lossfunc(outputs, labels)
        loss.backward()
        optimizer.step()
        Scheduler.step()

        corr += (outputs.argmax(1) == labels).sum().item()
        tot += labels.size(0)
        
        history.append(loss.item())
        
        # print(loss.item())
    print(i)
    t1 = (time.perf_counter() - t0) * 1000
    print(f"{t1:.3f}")
    accuracy.append(corr/tot)
    i += 1

model.eval()
with torch.no_grad():
    corr = 0
    tot = 0
    output = model(X_test)
    
    t0 = time.perf_counter()
    test_acc = (output.argmax(1) == Y_test).float().mean().item() * 100
    accuracy_test = test_acc

print(test_acc)
