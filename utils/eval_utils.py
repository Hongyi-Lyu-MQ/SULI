import torch

def evaluate(model, test_loader, device='cuda'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

def evaluate_instance_model_accuracy(model, test_loader, forget_loader, retain_loader, device):
    test_acc = evaluate(model, test_loader, device)
    forget_acc = evaluate(model, forget_loader, device)
    retain_acc = evaluate(model, retain_loader, device)
    print(f"Test Loader Accuracy: {test_acc:.2f}%")
    print(f"Forget Loader Accuracy: {forget_acc:.2f}%")
    print(f"Retain Loader Accuracy: {retain_acc:.2f}%")
