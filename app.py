
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# FastAPI application
app = FastAPI(
    title="Student Study Assistant Agent",
    description="AI assistant for student learning and exam preparation",
    version="1.0"
)


# Request model
class StudyRequest(BaseModel):
    mode: str = Field(
        description="Choose ask, explain, or notes"
    )
    topic: str = Field(
        min_length=1,
        max_length=2000
    )


# Initialize Gemma
def generate_with_gemma(prompt_text):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt_text
    )

    if not response.text:
        raise ValueError("Gemma returned an empty response")

    return response.text


# Home route
@app.get("/")
def home():

    return {
        "message": "Welcome to Student Study Assistant Agent!",
        "model": "Gemma 4 26B A4B",
        "features": [
            "Answer student questions",
            "Explain difficult topics",
            "Generate short notes"
        ],
        "endpoint": "/study",
        "documentation": "/docs"
    }


# Health check
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Student Study Assistant is running"
    }


# Main study assistant endpoint
@app.post("/study")
def study_assistant(request: StudyRequest):

    prompts = {

        "ask": """
        You are a helpful student study assistant.

        Answer the student's question correctly.
        Use simple English.
        Give examples wherever necessary.

        Student question: {topic}
        """,

        "explain": """
        You are a friendly teacher.

        Explain the given topic in very simple English.
        Explain step by step.
        Include an easy real-life example.

        Topic: {topic}
        """,

        "notes": """
        You are an exam preparation assistant.

        Generate short and easy study notes.

        Include:
        1. Definition
        2. Important points
        3. Example
        4. Conclusion

        Use simple English.

        Topic: {topic}
        """
    }

    if request.mode not in prompts:
        raise HTTPException(
            status_code=400,
            detail="Invalid mode. Choose ask, explain, or notes."
        )

    try:

        prompt = ChatPromptTemplate.from_template(
            prompts[request.mode]
        )

        # LangChain pipeline
        chain = (
            prompt
            | RunnableLambda(generate_with_gemma)
        )

        result = chain.invoke({
            "topic": request.topic
        })

        return {
            "agent": "Student Study Assistant",
            "model": "Gemma 4 26B A4B",
            "mode": request.mode,
            "topic": request.topic,
            "response": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
