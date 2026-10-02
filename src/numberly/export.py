import onnxruntime as ort
import torch

from numberly.predict import load_model


def main() -> None:
    model = load_model("optimal_model.pt")

    example_input = torch.zeros(1, 784)

    torch.onnx.export(
        model,
        (example_input,),
        "numberly.onnx",
        input_names=["input"],
        output_names=["logits"],
        external_data=False,
    )

    print("Exported to numberly.onnx")

    session = ort.InferenceSession("numberly.onnx")
    test_input = torch.rand(1, 784)

    onnx_output = session.run(None, {"input": test_input.numpy()})[0]
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    print("Largest difference:", abs(onnx_output - torch_output).max())
