import torch
import torch.nn.functional as F
import time
import copy
from torch.utils.data import DataLoader

from utils.eval_utils import evaluate

# ===== SU：Soft Unlearning =====
def distill_with_soft_relabel(
    original_model, student_model, forget_loader, optimizer, forget_class_idxs,
    epochs=1, device='cuda', distill_temperature=4
):
    """
    SU: unlearning
    """
    original_model.to(device)
    student_model.to(device)
    if not isinstance(forget_class_idxs, list):
        forget_class_idxs = [forget_class_idxs]
    for epoch in range(epochs):
        student_model.train()
        for data, targets in forget_loader:
            data, targets = data.to(device), targets.to(device)
            with torch.no_grad():
                teacher_logits = original_model(data)
                for forget_class_idx in forget_class_idxs:
                    teacher_logits[:, forget_class_idx] = float('-inf')
                soft_labels = F.softmax(teacher_logits / distill_temperature, dim=1)
            student_logits = student_model(data)
            student_log_probs = F.log_softmax(student_logits / distill_temperature, dim=1)
            loss = F.kl_div(student_log_probs, soft_labels.detach(), reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(student_model, forget_loader, device)
        if acc == 0:
            break
    return student_model

# ===== SULI: Self-Unlearning Layered Iteration =====

def calculate_entropy(logits):
    probabilities = F.softmax(logits, dim=1)
    log_probabilities = F.log_softmax(logits, dim=1)
    entropy = -(probabilities * log_probabilities).sum(dim=1)
    return entropy

def sort_data_loader_by_entropy(teacher_model, data_loader, device, batch_size_per_loader=200):
    """
    sort data from high entropy to low
    return: dataLoader
    """
    data_with_entropy = []
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            logits = teacher_model(data)
            entropy = calculate_entropy(logits)
            for i in range(len(data)):
                data_with_entropy.append((data[i].cpu(), targets[i], entropy[i].item()))
    # sort samples by entropy
    data_with_entropy.sort(key=lambda x: x[2], reverse=True)
    loaders = []
    for i in range(0, len(data_with_entropy), batch_size_per_loader):
        batch = data_with_entropy[i:i+batch_size_per_loader]
        dataset = [(x[0], x[1]) for x in batch]  
        loader = DataLoader(dataset, batch_size=len(batch), shuffle=True)
        loaders.append(loader)
    return loaders

def SPA_Iteration_unlearning(teacher_model, student_model, forget_loader, optimizer, forget_class_idxs, epochs=1, device='cuda', distill_temperature=1):
    teacher_model.to(device)
    student_model.to(device)
    if not isinstance(forget_class_idxs, list):
        forget_class_idxs = [forget_class_idxs]
    for epoch in range(epochs):
        student_model.train()
        for data, targets in forget_loader:
            data, targets = data.to(device), targets.to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(data)
                for forget_class_idx in forget_class_idxs:
                    teacher_logits[:, forget_class_idx] = float('-inf')
                soft_labels = F.softmax(teacher_logits / distill_temperature, dim=1)
            student_logits = student_model(data)
            student_log_probs = F.log_softmax(student_logits / distill_temperature, dim=1)
            loss = F.kl_div(student_log_probs, soft_labels.detach(), reduction='batchmean')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return student_model

def SelfUnlearning_Layered_Iteration(
    teacher_model, sorted_loaders, forget_class_idxs, forget_loader,
    epochs=10, device='cuda', distill_temperature=1, lr=0.01
):
    """
    SULI process
    """
    current_teacher_model = copy.deepcopy(teacher_model).to(device)
    previous_accuracy = -1
    for epoch in range(epochs):
        for loader_index, loader in enumerate(sorted_loaders):
            state_before_update = copy.deepcopy(current_teacher_model.state_dict())
            optimizer = torch.optim.Adam(current_teacher_model.parameters(), lr=lr, weight_decay=0)
            current_teacher_model = SPA_Iteration_unlearning(
                current_teacher_model, current_teacher_model, loader, optimizer, forget_class_idxs, epochs=1,
                device=device, distill_temperature=distill_temperature
            )
            forget_test_accuracy = evaluate(current_teacher_model, forget_loader, device=device)
            if forget_test_accuracy > previous_accuracy and previous_accuracy != -1:
                # 回滚
                current_teacher_model.load_state_dict(state_before_update)
            else:
                previous_accuracy = forget_test_accuracy
            if forget_test_accuracy == 0:
                break
        else:
            continue
        break
    return current_teacher_model
