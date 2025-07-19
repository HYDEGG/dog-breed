from fastAPI import FastAPI, HTTPException
from pydantic import BaseModel
from Backend.model.predictor import DogBreedPredictor

app = FastAPI()
predictor = DogBreedPredictor()

class InputData(BaseModel):
    image_path: str

@app.post("/predict")
def predict(input_data: InputData):
    try:
        prediction = predictor.predict(input_data.image_path)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        return {"error": "An error occurred during prediction."}