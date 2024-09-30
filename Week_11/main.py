from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Keep the existing sentiment analyzer for comparison
sentiment_analyzer = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/analyze-sentiment/", response_model=SentimentResponse)
def analyze_sentiment(request: TextRequest):
    result = sentiment_analyzer(request.text)[0]
    sentiment = result['label']
    confidence = result['score']

    return SentimentResponse(sentiment=sentiment, confidence=confidence)

@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)