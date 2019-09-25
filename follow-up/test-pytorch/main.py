import torch
import torch.nn as nn
import torch.autograd as autograd

# A model has a from of ``y = a*x + b``
model = nn.Linear(1, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

print(model.state_dict())

for epoch in range(2):
    for i, x in enumerate(torch.linspace(-1, 1, 1000)):
        x = x.unsqueeze(0)
        x.requires_grad_(True)

        optimizer.zero_grad()

        y = model(x)
        grad = autograd.grad(y, x, create_graph=True)[0]
        loss = torch.norm(grad - 2)**2
        loss.backward()
        optimizer.step()

print(model.state_dict())
