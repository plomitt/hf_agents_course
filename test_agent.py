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

from common import SubmitFinalAnswer, detect_media_type, MODELS, get_model
from image_agent import create_image_agent
from audio_agent import create_audio_agent
from reasoning_agent import code_to_text, csv_to_text, media_agent_tool

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
reasoning_agent_model_id="qwen/qwen3-next-80b-a3b-instruct"

reasoning_agent = Agent(
    # model=OpenRouter(
    #     api_key=openrouter_api_key,
    #     id=reasoning_agent_model_id,
    #     temperature=0.0,
    # ),
    model=get_model('local_oss'),
    instructions=[
        # "Always use the 'SubmitFinalAnswer' tool with your final answer.",
        # "Do not explain tool calls in text.",
        # "Call the 'SubmitFinalAnswer' to submit the answer, and aftwerwards respond with 'submitted'.",
    ],
    tools=[],
    reasoning=False,
    markdown=False,
)

# reasoning_agent.print_response("How many engines do boats typically have? task_id='12345'", show_tool_calls=True)
reasoning_agent.print_response('reply "banana"')