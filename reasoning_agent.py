from agno.agent import Agent, RunOutput 
from agno.media import Image
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv
import pathlib
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
import os

from image_agent import create_media_agent


load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
reasoning_agent_model_id="qwen/qwen3-coder-30b-a3b-instruct"

def media_agent_tool(media_path: str = "", question: str = "") -> str:
    """Use this tool to ask a specific question about a local media file (image, audio, or video).
    
    Args:
        media_path: The exact absolute local path to the media file.
        question: The specific question to ask about the media.
    
    Returns:
        str: A concise answer to the question, or 'idk' if not known, or 'file not found' if the file doesn't exist.
    """

    print(f"MediaProcessing received media_path: {media_path}")
    print(f"MediaProcessing received question: {question}")

    if not media_path or not pathlib.Path(media_path).exists():
        return 'file not found'
    
    media_path = str(media_path)
    print(f"MediaProcessing received media_path: {media_path}")

    media_agent = create_media_agent()
    return media_agent.run(question, images=[Image(filepath=media_path)])

def create_reasoning_agent():
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()

    reasoning_agent = Agent(
        model=OpenRouter(
            api_key=openrouter_api_key,
            id=reasoning_agent_model_id,
            temperature=0.0,
        ),
        instructions=system_prompt,
        tools=[DuckDuckGoTools(), WikipediaTools(), ArxivTools(), media_agent_tool],
        reasoning=True,
        markdown=True,
    )

    return reasoning_agent


if __name__ == "__main__":
    reasoning_agent = create_reasoning_agent()
    # reasoning_agent.print_response("SHow many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.", stream=True, show_full_reasoning=True)
    output: RunOutput = reasoning_agent.run("""How many queens are on the chess board??? image_paths=["media/chess.jpg"]""", stream=False)
    print(output.content)
