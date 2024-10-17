import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto" if device == "cuda" else None)

if device == "cpu":
    model = model.float()  # Ensure model is in float32 for CPU inference

class InferenceInput(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 32

@app.post("/inference")
async def inference(input_data: InferenceInput):
    try:
        inputs = tokenizer.apply_chat_template(input_data.messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        outputs = model.generate(inputs, max_new_tokens=input_data.max_new_tokens)
        text = tokenizer.batch_decode(outputs)[0]
        
        return {
            "generated_text": text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Phi-3-mini LLM Inference API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
