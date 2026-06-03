import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split


def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs=10):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            output, aux1, aux2 = outputs
            loss = criterion(output, labels) + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
        else:
            output = outputs
            loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 500 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Step {batch_idx+1}/{len(loader)} | Loss {running_loss / total:.4f}")

    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        if isinstance(outputs, labels):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        pred = outputs.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def train(model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    train_loss, val_loss, train_accuracy, val_accuracy = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(1, num_epochs+1):
        t_loss, t_acc = train_epoch(
            model, 
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs
        )
        v_loss, v_acc = train_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            epoch,
            num_epochs
        )

        scheduler.step()

        train_loss.append(t_loss)
        train_accuracy.append(t_acc)
        val_loss.append(v_loss)
        val_accuracy.append(v_acc)

        print(f"Train loss: {t_loss:.4f} | Train accuracy: {t_acc:.4f}")
        print(f"Val loss: {v_loss:.4f} | Val accuracy: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), 'GoogLeNet_best.pth')
            print(f"New best model saved with val accuracy {best_val_acc:.4f}")

    return train_loss, val_loss, train_accuracy, val_accuracy

def get_dataloaders(batch_size=32):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    train_set = datasets.CIFAR100('./data', train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100('./data', train=False, download=True, transform=test_transform)
    train_data, val_data = random_split(train_set, [45000, 5000])

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader, val_loader
    