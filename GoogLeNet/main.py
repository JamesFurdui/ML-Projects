import torch
import torch.nn as nn
import torch.optim as optim

from training_helpers import train_epoch, train, get_dataloaders, evaluate
from helper_layers import InceptionLayer, aux_classifier
from googleNetArchitecture import GoogLeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    model = GoogLeNet(num_classes=100, aux_logits=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
        )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    train_loss, val_loss, train_acc, val_acc = train(
        model,
        train_loader,
        test_loader,
        None,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=10
    )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    torch.save(model.state_dict(), "GoogLeNet_final.pth")
    print("Final model saved: GoogLeNet_final.pth")

if __name__ == "__main__":
    main()
