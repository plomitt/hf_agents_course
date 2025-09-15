# https://github.com/esshka/simple-react-agent/tree/main
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import sys

from common import _split_transcript_and_final, csv_to_text, code_to_text, get_final_answer, get_question_prompt_with_media, make_calc_tool, make_fetch_page_tool, make_web_search_tool
from image_agent import image_question_tool
from audio_agent import audio_question_tool

from react_agent import ReActAgent, ToolSpec
from openrouter_client import OpenRouterClient

load_dotenv()

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

def create_tools() -> list[ToolSpec]:
    tools = [
        make_web_search_tool("us-en", None, 8),
        make_fetch_page_tool(5000),
        make_calc_tool(),
        ToolSpec(
            name="csv_to_text",
            description="Use this tool to convert a CSV file to a text representation.",
            handler=csv_to_text,
            parameters= {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The exact relative local path to the CSV file."}
                },
                "required": ["file_path"],
                "additionalProperties": False,
            }
        ),
        ToolSpec(
            name="code_to_text",
            description="Use this tool to convert a code file to a text representation.",
            handler=code_to_text,
            parameters= {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The exact relative local path to the code file."}
                },
                "required": ["file_path"],
                "additionalProperties": False,
            }
        ),
        ToolSpec(
            name="image_question_tool",
            description="Use this tool to ask a specific question about an image file.",
            handler=image_question_tool,
            parameters= {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "The exact relative local path to the image file."},
                    "question": {"type": "string", "description": "The specific question to ask about the image."}
                },
                "required": ["image_path", "question"],
                "additionalProperties": False,
            }
        ),
        ToolSpec(
            name="audio_question_tool",
            description="Use this tool to ask a specific question about an audio file.",
            handler=audio_question_tool,
            parameters= {
                "type": "object",
                "properties": {
                    "audio_path": {"type": "string", "description": "The exact relative local path to the audio file."},
                    "question": {"type": "string", "description": "The specific question to ask about the audio."}
                },
                "required": ["audio_path", "question"],
                "additionalProperties": False,
            }
        ),
    ]

    return tools

def add_tools_to_agent(agent: ReActAgent, tools: list[ToolSpec]) -> ReActAgent:
    for tool in tools:
        agent.add_tool(tool)
    
    return agent

def create_react_agent():
    client = OpenRouterClient(app_name="hf_agents_course")
    agent = ReActAgent(
        client=client,
        model='google/gemini-2.5-flash',
        system_prompt=None,
        keep_history=True,
        temperature=0.1,
        max_rounds=8,
        max_tool_iters=8,
        reasoning_effort='high',
        parallel_tool_calls=False,
    )
    
    tools = create_tools()
    agent_with_tools = add_tools_to_agent(agent, tools)

    return agent_with_tools

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    q = get_question_prompt_with_media("cca530fc-4052-43b2-b130-b30968d8aa44", "what is the image about?",)
    agent = create_react_agent()
    answer = agent.ask(q)
    transcript, final = _split_transcript_and_final(answer.content)
    print(f'final> {final}')
    print(f'transcript> {transcript}')

# 89.71