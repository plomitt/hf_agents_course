from agno.agent import Agent, RunOutput 
from agno.media import Image, Audio
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv
from pathlib import Path
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools
import pandas as pd
import os

from common import SubmitFinalAnswer, detect_media_type
from image_agent import create_image_agent
from audio_agent import create_audio_agent


load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
reasoning_agent_model_id="qwen/qwen3-next-80b-a3b-instruct"

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
        str: A concise answer to the question, or 'idk' if not known, or 'file not found' if the file doesn't exist.
    """

    print(f"MediaProcessing received question: {question}")

    if not media_path or not Path(media_path).exists():
        return 'file not found'
    
    media_path = get_absolute_path(media_path)
    print(f"MediaProcessing received media_path: {str(media_path)}")

    media_type = detect_media_type(str(media_path))

    if media_type == 'image':
        media_agent = create_image_agent()
        return media_agent.run(question, images=[Image(filepath=media_path)])
    if media_type == 'audio':
        media_agent = create_audio_agent()
        return media_agent.run(question, audio=[Audio(filepath=media_path)])
    
    return 'file type not supported'

def create_reasoning_agent():
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()

    reasoning_agent = Agent(
        model=OpenRouter(
            api_key=openrouter_api_key,
            id=reasoning_agent_model_id,
            temperature=0.0,
        ),
        instructions=[
            "You are a concise and precise question answering agent.",
            "The final answer should always be in the shortest possible form: a single number, a single word, or at most a few words; never explain or elaborate unless explicitly asked; omit punctuation at the end of sentences/words, and omit any units unless explicitly asked; if the answer is a list of items, it should be a comma separated list."
            "Do not explain tool calls in text.",
            "Always call the 'SubmitFinalAnswer' to submit the final answer, and aftwerwards respond with 'submitted'.",
        ],
        tools=[DuckDuckGoTools(), WikipediaTools(), ArxivTools(), media_agent_tool, csv_to_text, code_to_text, SubmitFinalAnswer],
        reasoning=False,
        markdown=False,
    )

    return reasoning_agent


if __name__ == "__main__":
    reasoning_agent = create_reasoning_agent()
    # output: RunOutput = reasoning_agent.run("""How many rooks are on the chess board??? image_paths=["media_files/cca530fc-4052-43b2-b130-b30968d8aa44.png"] task_id='12345'""", stream=False)
    output: RunOutput = reasoning_agent.print_response("""How many engines do submarines typically have? task_id='12345'""", show_tool_calls=True)
    # print(output.content)
    # reasoning_agent.print_response("""How many engines do jets typically have? task_id='12345'""")
