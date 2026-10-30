
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal
from google import genai

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


app = FastAPI(
    title="Student Study Assistant Agent",
    description="AI assistant for students",
    version="1.0"
)


# Student input
class StudyRequest(BaseModel):
    mode: Literal["ask", "explain", "notes"]
    topic: str = Field(min_length=1, max_length=2000)


# Connect to Gemma
def generate_with_gemma(prompt_text):

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in Render")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt_text
    )

    if not response.text:
        raise ValueError("Gemma returned an empty response")

    return response.text


# Home page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Student Study Assistant</title>
    </head>
    <body style="font-family:Arial;text-align:center;padding:50px">
        <h1>Student Study Assistant</h1>
        <p>Your AI learning assistant powered by LangChain and Gemma.</p>
        <a href="/agent/playground/">Open AI Assistant</a>
        <br><br>
        <a href="/docs">API Documentation</a>
    </body>
    </html>
    """


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Student Study Assistant is running"
    }


# Interactive playground
@app.get("/agent/playground/", response_class=HTMLResponse)
def playground():

    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Student Study Assistant</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                margin: 0;
                padding: 25px;
            }

            .container {
                max-width: 700px;
                margin: 30px auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 15px #ddd;
            }

            h1 {
                color: #2563eb;
                text-align: center;
            }

            p {
                text-align: center;
                color: #555;
            }

            select, textarea, button {
                width: 100%;
                padding: 13px;
                margin-top: 15px;
                box-sizing: border-box;
                border-radius: 7px;
                font-size: 15px;
            }

            select, textarea {
                border: 1px solid #ccc;
            }

            textarea {
                height: 120px;
                resize: vertical;
            }

            button {
                background: #2563eb;
                color: white;
                border: none;
                cursor: pointer;
            }

            button:disabled {
                opacity: 0.6;
                cursor: wait;
            }

            #result {
                margin-top: 20px;
                padding: 18px;
                background: #eff6ff;
                border-radius: 8px;
                white-space: pre-wrap;
                line-height: 1.6;
                display: none;
            }

            .links {
                text-align: center;
                margin-top: 20px;
            }
        </style>
    </head>

    <body>
        <div class="container">

            <h1>Student Study Assistant 🎓</h1>

            <p>Learn better with your AI study assistant!</p>

            <label>Select your requirement:</label>

            <select id="mode">
                <option value="ask">Ask a Question</option>
                <option value="explain">Explain a Topic</option>
                <option value="notes">Generate Short Notes</option>
            </select>

            <label>Enter your question or topic:</label>

            <textarea id="topic"
                placeholder="Example: Explain Python programming with an example">
            </textarea>

            <button id="submitBtn" onclick="askAgent()">
                Generate Answer
            </button>

            <div id="result"></div>

            <div class="links">
                <a href="/docs">API Documentation</a>
            </div>

        </div>

        <script>
            async function askAgent() {

                const mode = document.getElementById("mode").value;
                const topic = document.getElementById("topic").value.trim();
                const result = document.getElementById("result");
                const button = document.getElementById("submitBtn");

                if (!topic) {
                    alert("Please enter a topic or question.");
                    return;
                }

                button.disabled = true;
                button.textContent = "Generating...";
                result.style.display = "block";
                result.textContent = "Please wait. Your AI assistant is preparing an answer...";

                try {

                    const response = await fetch("/study", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            mode: mode,
                            topic: topic
                        })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || "Request failed");
                    }

                    result.textContent = data.response;

                } catch (error) {

                    result.textContent = "Error: " + error.message;

                } finally {

                    button.disabled = false;
                    button.textContent = "Generate Answer";
                }
            }
        </script>

    </body>
    </html>
    """


# AI study endpoint
@app.post("/study")
def study_assistant(request: StudyRequest):

    prompts = {

        "ask": """
        You are a helpful student study assistant.
        Answer the question correctly using simple English.
        Give examples wherever useful.

        Student question: {topic}
        """,

        "explain": """
        You are a friendly teacher.
        Explain the topic step by step using simple English.
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

    try:

        prompt = ChatPromptTemplate.from_template(
            prompts[request.mode]
        )

        # LangChain prompt-to-Gemma pipeline
        chain = prompt | RunnableLambda(
            lambda prompt_value: generate_with_gemma(
                prompt_value.to_string()
            )
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
