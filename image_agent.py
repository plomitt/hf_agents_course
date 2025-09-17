from pathlib import Path
from agno.agent import Agent
from agno.media import Image

from common import detect_media_type, get_model, get_absolute_path, get_media_path

def create_image_agent():
    media_agent = Agent(
        model=get_model('open_dusk_alpha'),
        instructions="You are a concise image question-answering assistant. You are given a question and an image file. You need to answer the question based on the image file.",
    )
    
    return media_agent

def image_question_tool(args):
    """Use this tool to ask a specific question about an image file.
    
    Args:
        image_path: The exact relative local path to the image file.
        question: The specific question to ask about the image.
    
    Returns:
        str: The answer to the question, or 'idk', or 'file not found', or 'file type not supported'.
    """
    image_path = args.get("image_path", None)
    question = args.get("question", None)

    print(f"image_tool received question: {question[:50]}...")

    if not image_path or not Path(image_path).exists():
        return 'file not found'
    
    image_path = get_absolute_path(image_path)
    print(f"image_tool received image_path: {str(image_path)}")

    media_type = detect_media_type(str(image_path))

    if media_type == 'image':
        image_agent = create_image_agent()
        answer = image_agent.run(question, images=[Image(filepath=image_path)])
        print(f"image_tool received answer: {answer.content}")
        return answer.content
    
    return 'file type not supported'

if __name__ == "__main__":
    image_agent = create_image_agent()
    image_agent.print_response("Represent the chess board on the image as a 2D table",  images=[Image(filepath=get_media_path('image1'))], stream=True, show_tool_calls=True, show_full_reasoning=True)
