import os
import gradio as gr
from google import genai


# Get Gemini API key from Render Environment Variables
API_KEY = os.environ.get("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.6-flash"


def study_assistant(message, history):

    if not message.strip():
        return "Please enter a question."

    prompt = f"""
You are an Interactive Study Assistant.

Answer the student's question clearly and briefly.

Rules:
- Use simple language.
- Keep the answer short.
- Use multiple lines when needed.
- Give an example when useful.
- Do not provide unnecessary information.

Student Question:
{message}
"""

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"


demo = gr.ChatInterface(
    fn=study_assistant,
    title="Interactive Study Assistant",
    description="Ask questions and learn with Gemini AI.",
    examples=[
        "What is Artificial Intelligence?",
        "Explain Machine Learning.",
        "What is a primary key?",
        "Explain a neural network."
    ]
)


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
