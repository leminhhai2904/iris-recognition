import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.models import DeepIrisResNet
from src.dataloader import IITDIrisDataset


def create_stratified_splits(dataset, test_samples=4, val_samples=1):
    """
    Groups data by subject to ensure exact counts per class are split.
    """
    class_indices = defaultdict(list)
    for idx, (_, label, _) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        # 4 images are used as test samples randomly, and the rest are using for training and validation.
        test_indices.extend(indices[:test_samples])
        val_indices.extend(indices[test_samples : test_samples + val_samples])
        train_indices.extend(indices[test_samples + val_samples :])

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = IITDIrisDataset(transform=transform)

    # 4 for test, 1 for validation, 5 for training (per person)
    train_dataset, val_dataset, test_dataset = create_stratified_splits(
        dataset, test_samples=4, val_samples=1
    )

    train_loader = DataLoader(
        train_dataset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=24, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=24, shuffle=False, num_workers=2, pin_memory=True
    )

    model = DeepIrisResNet(num_classes=224).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)

    epochs = 100
    best_val_acc = 0.0

    # Initialize metrics logging
    metrics_file = open("checkpoints/metrics.csv", "w", newline="")
    csv_writer = csv.writer(metrics_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}"
        )

        # Write metrics to disk
        csv_writer.writerow(
            [
                epoch + 1,
                epoch_train_loss,
                epoch_train_acc,
                epoch_val_loss,
                epoch_val_acc,
            ]
        )
        metrics_file.flush()

        # Save model checkpoints
        torch.save(model.state_dict(), "checkpoints/latest_model.pth")
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("--> Saved new best model")

    metrics_file.close()

    # Testing Phase
    print("\nStarting evaluation on test set...")

    # Load the best weights found during validation
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    train_model()
