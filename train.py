import torch
import copy
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

def plot_train_losses(model_type):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='train_losses')
    plt.plot(val_losses, label='val_losses')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(model_type+'_Train_Val_Loss'+'.png')  # 保存为PNG格式

def train_val_model(model, train_loader, val_loader, optimizer, criterion, device, model_type, epochs, path_model_param):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total_acc = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_acc += target.size(0)
            correct += (predicted == target).sum().item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_acc = correct / total_acc
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Accuracy_train: {train_acc*100:.4f}%')

        model.eval()
        val_loss = 0
        correct = 0
        total_acc = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_acc += target.size(0)
                correct += (predicted == target).sum().item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_acc = correct / total_acc
        print(f'Validation Loss: {val_loss:.4f}, Accuracy_val: {val_acc*100:.4f}%')
        # 保存最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts, path_model_param)
    print(f"Model saved to {path_model_param}")
    print('Best accuracy: {:.4f}%'.format(100 * best_acc))
    
    # 调用绘图函数
    plot_train_losses(model_type)



