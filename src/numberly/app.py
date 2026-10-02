from io import BytesIO

from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import transforms

from numberly.predict import load_model, predict

app = FastAPI()
model = load_model()
to_tensor = transforms.ToTensor()


@app.post("/predict")
async def predict_digit(file: UploadFile):
    contents = await file.read()

    image = Image.open(BytesIO(contents)).convert("L").resize((28, 28))
    tensor = to_tensor(image)

    prediction, probabilities = predict(model, tensor)

    return {"prediction": prediction, "probabilities": probabilities.tolist()}
