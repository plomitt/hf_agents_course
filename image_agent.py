from agno.agent import Agent
from agno.media import Image

from common import get_model, get_media_path

def create_image_agent():
    media_agent = Agent(
        model=get_model('open_dusk_alpha'),
        instructions="You are a concise media question-answering assistant. You are given a question and a media file. You need to answer the question based on the media file.",
    )
    
    return media_agent

if __name__ == "__main__":
    media_agent = create_image_agent()
    # media_agent.print_response("Represent the chess board on the image as a 2D table",  images=[Image(filepath=get_media_path('image1'))], stream=True, show_tool_calls=True, show_full_reasoning=True)
