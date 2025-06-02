import torch
import torch.nn as nn
import torch.nn.functional as F

def train_model(model, train_loader, epochs=40, max_lr=0.01, grad_clip=0.1, weight_decay=0, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
    return model
