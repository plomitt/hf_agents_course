import os
from os import getenv
from pathlib import Path
import pathlib
import time
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.openrouter import OpenRouter
from agno.media import Image, Audio, Video, File
from agno.tools.wikipedia import WikipediaTools
from agno.tools.arxiv import ArxivTools

from agno.tools import tool
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_model_id = "openai/gpt-5-mini"
image_agent_model_id="openai/gpt-5-mini"
# openrouter_model_id = "google/gemini-2.5-flash"
# openrouter_model_id = "google/gemini-2.5-flash-lite"
# openrouter_model_id = "openrouter/sonoma-sky-alpha"
# openrouter_model_id = "moonshotai/kimi-k2-0905"

media_agent = Agent(
    model=OpenRouter(
        api_key=openrouter_api_key,
        id=openrouter_model_id,
        temperature=0.05,
    ),
    reasoning=False,
    system_message=(
        "You are a concise media question-answering assistant. "
        "When the user provides or references media, your sole task is to answer the specific question asked about it. "
        "Do not provide a general description. "
        "If the question can be answered from the media, provide the most direct, concise answer possible. "
        "If the media contains text content, transcribe or summarize it as needed to answer the question. "
        "If the question cannot be answered from the media, reply 'idk', do not make up an answer. "
        "Do not elaborate or add extra information. "
    )
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
CODE_EXTS = {".py", ".js", ".json", ".html", ".css", ".java", ".c", ".cpp", ".cs", ".rb", ".go", ".rs", ".ts"}
SPREADSHEET_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}
TEXT_EXTS = {".txt", ".md", ".pdf", ".docx"}

def _detect_media_type(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in CODE_EXTS:
        return "code"
    if ext in SPREADSHEET_EXTS:
        return "spreadsheet"
    if ext in TEXT_EXTS:
        return "text"

    return "file"

def _call_media_agent(media_type, media_path, prompt):
    images=[Image(filepath=media_path)] if media_type == 'image' else None
    audio=[Audio(filepath=media_path)] if media_type == 'audio' else None
    videos=[Video(filepath=media_path)] if media_type == 'video' else None


    # print(f"Calling media_agent with prompt: {prompt}")
    # print(f"Calling media_agent with media_type: {media_type}")
    # print(f"with images: {images}")
    # print(f"with audio: {audio}")
    # print(f"with videos: {videos}")
    # print(f"with files: {files}")

    return media_agent.run(prompt, images=images, audio=audio, videos=videos)


def MediaQuestionTool(media_path: str = "", question: str = "") -> str:
    """Use this tool to ask a specific question about a local media file (image, audio, or video).
    
    Args:
        media_path: The exact absolute local path to the media file.
        question: The specific question to ask about the media.
    
    Returns:
        str: A concise answer to the question, or 'idk' if not known, or 'file not found' if the file doesn't exist.
    """

    if not media_path or not pathlib.Path(media_path).exists():
        return 'file not found'
    
    media_path = str(media_path)
    print(f"MediaProcessing received media_path: {media_path}")

    media_type = _detect_media_type(media_path)

    # Build the prompt for media_agent using the convention your media_agent expects:
    prompt = f"Describe the content of this {media_type}, including information relevant to the following user question. " + question

    # Call media_agent
    resp = _call_media_agent(media_type, media_path, prompt)
    result = resp.content
    print(f"MediaProcessing got response: {result}")

    return result

reasoning_agent = Agent(
    # model=OpenAILike(
    #     id="openai/gpt-oss-20b",
    #     api_key="123",
    #     base_url="http://192.168.1.157:1234/v1",
    #     reasoning_effort="high",
    #     temperature=0.05,
    # ),
    model=OpenRouter(
        api_key=openrouter_api_key,
        id=openrouter_model_id,
        temperature=0.05,
    ),
    tools=[GoogleSearchTools(), WikipediaTools(), ArxivTools(), MediaQuestionTool],
    reasoning=False,
    system_message=(
        "You are a concise assistant that uses tools to answer questions."
        "Before answering any question, first think deeply and step by step about what tools you should use to find the answer."
        "Use the 'GoogleSearchTools' tool to find relevant information."
        "ALWAYS USE 'GoogleSearchTools', for every question."
        "Use the 'MediaQuestionTool' tool to ask a specific question about an image, audio, or video file. You can call this tool multiple times with different questions about the same media file if needed."
        "ALWAYS USE 'MediaQuestionTool' when the user provides or references an image, audio, or video files in his question."
        "When asked about a media file, you will receive a path to the local file in the prompt, which you should pass exactly as it is to 'MediaQuestionTool' along with your specific question about it."
        "IMPORTANT: Always provide a specific question to 'MediaQuestionTool' that is relevant to the user's original question, rather than just asking for a general description of the media."
        "IMPORTANT: Always provide the exact local path as-is to 'MediaQuestionTool' without modifying it."
        "IMPORTANT: Currenly video files are not supported by 'MediaQuestionTool', so when asked about video just reply 'video not supported'."
        "Use the 'WikipediaTools' tool to get summaries of topics from Wikipedia."
        "Use the 'ArxivTools' tool to get summaries of academic papers from arXiv."
        "IMPORTANT: Always use tools to find information, do not try to answer from memory."
        "IMPORTANT: ALways use tools in this order: WikipediaTools, ArxivTools, GoogleSearchTools, if the question is not about media."
        "IMPORTANT: Always use tools in this order: WikipediaTools, ArxivTools, GoogleSearchTools, MediaQuestionTool, if the question is about media. And remember to call MediaQuestionTool as many times as needed."
        "After calling all the needed tools, think step by step about the information you get from the tools, and how it relates to the question, then formulate your final answer."
        "Do not just copy and paste the content from the tools; instead, synthesize the information to provide a concise and accurate answer."
        "Think step by step about the information you get from the tools, and how it relates to the question, then formulate your final answer."
        "Always answer in the shortest possible form:"
        "a single number, a single word, or at most a few words."
        "Never explain or elaborate unless explicitly asked."
        "Omit punctuation at the end of sentences/words, and omit any units unless explicitly asked."
        "If you don't know, reply 'idk', do not try to make up an answer. "
    )
)

def _handle_paused_run(agent: Agent, prompt: str) -> str:
    print(f"Agent input: {prompt}")

    start = time.time()
    run_response = agent.run(prompt)

    # Set a counter for the loop depth
    loop_count = 0
    loop_limit = 3

    # Loop until the agent run is no longer paused or the loop depth is reached
    while run_response.is_paused and loop_count < loop_limit:
        print(f"Agent run is paused. Handling tool call {loop_count + 1}...")
        
        # Iterate through tool calls and confirm them
        for tool in run_response.tools:
            if tool.requires_confirmation:
                tool.confirmed = True
                
        # Continue the agent's run with the confirmed tool call
        run_response = agent.continue_run()
        loop_count += 1

    end = time.time()
        
    if not run_response.is_paused:
        # The agent successfully completed the task
        final_response = run_response
        final_response_content = final_response.content

        print(f"Agent processing time: {end - start:.2f} seconds")
        print(f"Agent loop amount: {loop_count + 1}")
        print(f"Agent returning new answer: {final_response_content}")
        return final_response_content
    else:
        # The loop reached its depth limit, and the agent is still paused
        print(f"AGENT ERROR: Agent is still paused after {loop_limit} iterations.")
        return "AGENT ERROR: Unable to complete the task due to unconfirmed tool calls."


# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str, media_path: str = None) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        if media_path and Path(media_path).exists():
            media_type = _detect_media_type(media_path)

            if media_type not in {'image', 'audio', 'video'}:
                with open(media_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    agent_input = f"Attached {media_type} file content:\n{file_content}\n\nQuestion: {question}"
            else:
                # Format the input so the reasoning agent sees the media reference and is likely to call MediaProcessing
                agent_input = f"Media: {media_type}={media_path} | Question: {question}"
        else:
            agent_input = question

        new_answer_content = _handle_paused_run(reasoning_agent, agent_input)

        return new_answer_content


# --- Example Calls ---

# media_files = {
#     'audio1': 'media_files/1f975693-876d-457b-a649-393859e79bf3.mp3',
#     'table1': 'media_files/7bd855d8-463d-4ed5-93ca-5fe35145f733.csv',
#     'video1': 'media_files/9d191bce-651d-4746-be2d-7ef8ecadb9c2.mp4',
#     'audio2': 'media_files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3',
#     'video2': 'media_files/a1e91b78-d3d8-4675-bb8d-62741b4b68a6.mp4',
#     'image1': 'media_files/cca530fc-4052-43b2-b130-b30968d8aa44.png',
#     'code1': 'media_files/f918266a-b3e0-4914-865d-4faa564f1aef.py'
# }

# agent_1 = BasicAgent()
# media_path = Path(__file__).parent.joinpath(media_files['video1'])
# answer = agent_1("What is this media about?", media_path=str(media_path))
# print(answer)