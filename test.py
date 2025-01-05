import torch
import os

def test_model(model, test_loader, criterion, device, path_model_param):
    model.load_state_dict(torch.load(path_model_param, map_location=device))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy: {:.4f}%'.format(100 * correct / total))
