import torch
from PIL import Image
from torchvision import transforms

to_tensor = transforms.ToTensor()


def preprocess(image: Image.Image) -> torch.Tensor | None:
    image = image.convert("L")

    # 1. crop to the drawing itself
    bbox = image.getbbox()
    if bbox is None:
        return None  # empty canvas, nothing drawn
    digit = image.crop(bbox)

    # 2. scale so the longest side is 20 pixels, keeping proportions
    width, height = digit.size
    scale = 20 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(new_size, Image.Resampling.LANCZOS)

    # 3. find the center of mass of the white pixels
    pixels = to_tensor(digit)[0]  # shape (height, width)
    total = pixels.sum()
    rows = torch.arange(pixels.shape[0], dtype=torch.float32)
    cols = torch.arange(pixels.shape[1], dtype=torch.float32)
    center_y = (pixels.sum(dim=1) * rows).sum() / total
    center_x = (pixels.sum(dim=0) * cols).sum() / total

    # 4. paste it on a black 28x28 image, so that the center of mass lands in the middle
    result = Image.new("L", (28, 28), 0)
    result.paste(digit, (round(14 - center_x.item()), round(14 - center_y.item())))

    return to_tensor(result)
