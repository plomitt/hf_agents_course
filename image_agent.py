import os
from agno.models.openrouter import OpenRouter
from agno.agent import Agent, RunOutput
from dotenv import load_dotenv
from agno.media import Image

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
image_agent_model_id="openrouter/sonoma-dusk-alpha"


def create_media_agent():
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
        self.agent = create_media_agent()
    
    def run(self, question: str, image_paths: list[str] = None) -> RunOutput:
        return self.agent.run(question, images=[Image(filepath=image_path) for image_path in image_paths])


if __name__ == "__main__":
    media_agent=MediaAgent()
    response: RunOutput = media_agent.run("Represent the chess board on the image as a 2D table", image_paths=["media/chess.jpg"])
    print(response.content)