import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer):
    total_loss = 0.0

    for images, labels in loader:
        flat = images.flatten(start_dim=1)

        output = model(flat)

        loss = F.cross_entropy(output, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loader)

    return average_loss


def evaluate(model, loader):
    total_correct_predictions = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            flat = images.flatten(start_dim=1)

            output = model(flat)

            loss = F.cross_entropy(output, labels)

            predicted = output.argmax(dim=1)
            correct = (predicted == labels).sum().item()

            total_correct_predictions += correct
            total_loss += loss.item()

    accuracy = (total_correct_predictions / len(loader.dataset)) * 100.0
    average_loss = total_loss / len(loader)

    return accuracy, average_loss
