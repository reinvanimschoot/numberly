from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.transforms.functional import to_pil_image

from numberly.model import build_model


def main() -> None:
    Path("mistakes").mkdir(exist_ok=True)

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    image, label = test_data[0]

    model = load_model()

    for i in range(len(test_data)):
        image, label = test_data[i]
        prediction, _probabilities = predict(model, image)

        if prediction != label:
            to_pil_image(image).save(
                f"mistakes/{i}_predicted_{prediction}_correct_{label}.png"
            )


def load_model(path="best_model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def predict(model, image):
    with torch.no_grad():
        flat = image.unsqueeze(0).flatten(start_dim=1)
        output = model(flat)

        prediction = output.argmax(dim=1).item()
        probabilities = torch.softmax(output, dim=1)

        return prediction, probabilities[0]
