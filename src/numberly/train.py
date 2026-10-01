import torch

from numberly.data import build_loaders
from numberly.model import build_model
from numberly.report import print_checkpoint, print_epoch_report, print_header
from numberly.training import evaluate, train_epoch


def main() -> None:
    model = build_model()
    training_loader, validation_loader = build_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print_header()

    min_validation_loss = float("inf")
    patience = 3
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, 21):
        train_loss = train_epoch(model, training_loader, optimizer)
        accuracy, validation_loss = evaluate(model, validation_loader)

        print_epoch_report(epoch, train_loss, accuracy, validation_loss)

        if validation_loss < min_validation_loss:
            epochs_without_improvement = 0
            min_validation_loss = validation_loss

            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print_checkpoint(best_epoch)
            break
