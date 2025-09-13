from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
from agno.media import Image, Audio
from agno.agent import Agent 
from pathlib import Path
import pandas as pd

from common import get_media_path, get_model, SubmitFinalAnswer, detect_media_type
from image_agent import create_image_agent
from audio_agent import create_audio_agent

def get_absolute_path(media_path: str) -> Path:
    try:
        return Path(__file__).parent.joinpath(media_path)
    except Exception as e:
        error = f"Error getting absolute path: {e}"
        print(error)
        return error

def csv_to_text(file_path: str) -> str:
    """Convert a CSV file to a text representation.

    Args:
        file_path: The path to the CSV file.

    Returns:
        str: The text representation of the CSV file.
    """
    try:
        path = get_absolute_path(file_path)
        df = pd.read_csv(path)
        return df.to_string()
    except Exception as e:
        return f"Error reading CSV file: {e}"

def code_to_text(file_path: str) -> str:
    """Convert a code file to a text representation.

    Args:
        file_path: The path to the code file.

    Returns:
        str: The text representation of the code file.
    """
    try:
        path = get_absolute_path(file_path)
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading code file: {e}"

def media_agent_tool(media_path: str = "", question: str = "") -> str:
    """Use this tool to ask a specific question about a local media file (image or audio).
    
    Args:
        media_path: The exact relative local path to the media file.
        question: The specific question to ask about the media.
    
    Returns:
        str: A concise answer to the question, or 'idk', or 'file not found', or 'file type not supported'.
    """

    print(f"media_agent_tool received question: {question[:50]}...")

    if not media_path or not Path(media_path).exists():
        return 'file not found'
    
    media_path = get_absolute_path(media_path)
    print(f"media_agent_tool received media_path: {str(media_path)}")

    media_type = detect_media_type(str(media_path))

    if media_type == 'image':
        media_agent = create_image_agent()
        return media_agent.run(question, images=[Image(filepath=media_path)])
    if media_type == 'audio':
        media_agent = create_audio_agent()
        return media_agent.run(question, audio=[Audio(filepath=media_path)])
    
    return 'file type not supported'

def create_reasoning_agent():
    with open("system_prompt_2.md", "r") as f:
        system_prompt = f.read()

    reasoning_agent = Agent(
        model=get_model('local_oss'),
        instructions=system_prompt,
        tools=[DuckDuckGoTools(), WikipediaTools(), ArxivTools(), media_agent_tool, csv_to_text, code_to_text, SubmitFinalAnswer],
        reasoning=False,
        markdown=False,
        use_json_mode=False
    )

    return reasoning_agent

if __name__ == "__main__":
    reasoning_agent = create_reasoning_agent()
    # reasoning_agent.print_response(f"""How many rooks are on the chess board??? image_paths=["{get_media_path('image1')}"] task_id='12345'""", stream=True, show_tool_calls=True, show_full_reasoning=True)
    reasoning_agent.print_response("""How many engines do cars typically have? task_id='12345'""", stream=True, show_tool_calls=True, show_full_reasoning=True)
