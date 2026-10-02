# Numberly

Numberly is an educational project I built while working through _Build a Large Language Model (From Scratch)_ by Sebastian Raschka.
I wanted to deeply understand how neural networks are built, trained, validated and deployed.

Numberly allows you to draw a digit, and a small neural network guesses which one it is. The trained model runs entirely in the browser, so the live app is a static site with no backend.

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/reinvanimschoot/numberly)**

## How it works

If you have no background in AI, I added an `annotated_example.py`, which was the file I started doodling in before comitting to a more fleshed out project. It contains most of my notes that helped me understand what was going on under the hood.

### The dataset

The network was built and trained with PyTorch on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset of handwritten digits.

### The model

A small fully connected network (multilayer perceptron) with about 100,000 parameters:

- Linear layer of 128 neurons, each accepting 784 inputs (28 x 28 grayscale image, flattened into 784 numbers)
- ReLU
- Linear layer of 10 neurons (the 10 possible classes), each accepting 128 inputs

The highest of the 10 output scores is the prediction.

### Training

- **Data:** 60,000 MNIST training images, in shuffled batches of 64
- **Loss:** cross-entropy
- **Optimizer:** Adam, learning rate 0.001
- **Validation:** after every epoch, on the 10,000 MNIST test images
- **Early stopping:** training stops when the validation loss hasn't improved for 3 epochs, and the best checkpoint is kept

The best model reaches about **97.7% accuracy** on the test set.

### Preprocessing

MNIST digits are cropped, scaled so their longest side is 20 pixels, and centered by center of mass in a 28 × 28 image. A digit drawn on a 280 × 280 canvas looks very different, so before predicting, the app applies the same steps to the drawing. Without this, accuracy on drawn digits drops sharply.

### Running in the browser

The trained PyTorch model is exported to [ONNX](https://onnx.ai/) and run in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). The model file (about 400 KB) is hosted in a Hugging Face model repository, and the page downloads it on load.

## Project structure

```
numberly/
├── src/numberly/
│   ├── data.py       # MNIST datasets and data loaders
│   ├── model.py      # the network definition
│   ├── training.py   # one training epoch, and validation
│   ├── train.py      # the training command (epochs, early stopping, checkpoint)
│   ├── predict.py    # loading the model and predicting test images
│   └── export.py     # exporting the model to ONNX
├── web/
│   ├── index.html    # the drawing app: canvas, preprocessing, inference
│   └── README.md     # configuration for the Hugging Face Space
└── pyproject.toml
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Train

```bash
uv run numberly-train
```

Downloads MNIST into `data/` on the first run, trains with early stopping, and saves the best weights to `best_model.pt`.

### Predict

```bash
uv run numberly
```

Loads `best_model.pt` and predicts a few images from the test set.

### Export

```bash
uv run numberly-export
```

Exports the model to `numberly.onnx`, and checks that its output matches the PyTorch model.

### Run the app locally

```bash
uv run python -m http.server 8000 --directory web
```

Then open [http://localhost:8000](http://localhost:8000). The page loads the model from Hugging Face, and falls back to a `numberly.onnx` in `web/` if that fails.

## Deployment

- **Model:** `numberly.onnx` is uploaded to the [reinvanimschoot/numberly](https://huggingface.co/reinvanimschoot/numberly) model repository:

  ```bash
  hf upload reinvanimschoot/numberly numberly.onnx
  ```

- **App:** the `web/` folder is deployed as a static Hugging Face Space, using a git subtree push:

  ```bash
  git subtree push --prefix web space main
  ```

Model files (`*.pt`, `*.onnx`) and the MNIST data are kept out of git.
