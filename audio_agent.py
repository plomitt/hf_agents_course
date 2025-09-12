import os
from agno.models.openrouter import OpenRouter
from agno.agent import Agent, RunOutput
from dotenv import load_dotenv
from agno.media import Audio

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
image_agent_model_id="google/gemini-2.5-flash"


def create_audio_agent():
    media_agent = Agent(
        model=OpenRouter(
            api_key=openrouter_api_key,
            id=image_agent_model_id,
            temperature=0.0,
        ),
        instructions="You are a concise media question-answering assistant. You are given a question and a media file. You need to answer the question based on the media file.",
    )
    
    return media_agent

class MediaAgent:
    def __init__(self):
        self.agent = create_audio_agent()
    
    def run(self, question: str, audio_paths: list[str] = None) -> RunOutput:
        return self.agent.run(question, audio=[Audio(filepath=audio_path) for audio_path in audio_paths])


if __name__ == "__main__":
    media_agent=MediaAgent()
    response: RunOutput = media_agent.run("What is the audio recording about?", audio_paths=["media_files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"])
    print(response.content)