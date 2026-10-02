from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from numberly.predict import load_model, predict
from numberly.preprocess import preprocess

app = FastAPI()
model = load_model()
to_tensor = transforms.ToTensor()


@app.post("/predict")
async def predict_digit(file: UploadFile):
    contents = await file.read()

    image = Image.open(BytesIO(contents)).convert("L").resize((28, 28))
    tensor = preprocess(image)

    if tensor is None:
        raise HTTPException(status_code=400, detail="Nothing drawn")

    prediction, probabilities = predict(model, tensor)

    return {"prediction": prediction, "probabilities": probabilities.tolist()}


app.mount("/", StaticFiles(directory="web", html=True), name="web")
