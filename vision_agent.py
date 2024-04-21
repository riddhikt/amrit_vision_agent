from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
from uagents import Model
from uagents.query import query
import google.generativeai as genai
from PIL import Image
from io import BytesIO
import requests
import uvicorn

compare_endpoint = "http://localhost:8000/compare-task"
AGENT_ADDRESS = "agent1q292k4f2c0d2snxd276ynfyshxd3quzzxj93n2ehftgrwkdhfavhgxsmu5z"


class TestRequest(Model):
    message: str


class AgentRequest(Model):
    image: str
    prompt: str


async def agent_query(req):
    response = await query(destination=AGENT_ADDRESS, message=req, timeout=15.0)
    data = json.loads(response.decode_payload())
    return data["text"]


app = FastAPI()
# Add CORS middleware to allow specific origins or all
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
genai.configure(api_key='AIzaSyANvgl8Qc8lW3bPOxtcFzj7yFgkjbPBxZE')
description_model = genai.GenerativeModel('models/gemini-pro-vision')


class ImageRequest(BaseModel):
    image_base64: str
    task: str


@app.post("/describe-image")
async def describe_image(request: ImageRequest):
    try:
        # Decode the image from base64 string
        image_data = base64.b64decode(request.image_base64.split(",")[1])  # Ensure proper split if data URL
        image = Image.open(BytesIO(image_data))

        # Generate description
        prompt = """Look at this image, and describe what is happening in the image, make sure you capture every 
        detail of it and return a paragraph containing the complete description"""
        response = description_model.generate_content([prompt, image])

        payload = {
            'description': response.text,
            'task': request.task
        }
        # Convert the dictionary to a JSON string
        headers = {'Content-Type': 'application/json'}

        # Send a POST request to the FastAPI server
        response = requests.post(compare_endpoint, json=payload, headers=headers)
        if "true" or "True" in response.text:
            return {"result": "true"}
        elif "false" or "false" in response.text:
            return {"result": "false"}
        else:
            return {"result": "some error occurred"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return "Hello from the Agent controller"


@app.post("/endpoint")
async def make_agent_call(req: AgentRequest):
    try:

        request_data = json.dumps({"image": req.image, "prompt": req.prompt})
        res = await agent_query(request_data)
        return {"message": "successful call - agent response", "response": res}
    except Exception as e:
        return {"message": "unsuccessful agent call", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run("vision_agent:app", host="0.0.0.0", port=8001, reload=True)
