import os
from os import getenv
from pathlib import Path
import pathlib
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.openrouter import OpenRouter
from agno.media import Image, Audio, Video, File
from agno.tools import tool
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_model_id = "google/gemini-2.0-flash-lite-001"

media_agent = Agent(
    model=OpenRouter(
        api_key=openrouter_api_key,
        id=openrouter_model_id,
        temperature=0.05,
    ),
    reasoning=True,
    system_message=(
        "You are a concise media description assistant. "
        "When the user provides or references media in his question, analyze the media and return its description/explaination/answer to the question. "
        "Explain what the media is about. "
        "Also include any text content or important/relevant information/facts that are present in the media. "
        "Always limit the final answer to a single paragraph, or, if possible and reasonable, to a sigle sentence. "
        "If you don't know, reply 'idk', do not try to make up an answer. "
    )
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

def _detect_media_type(path: str) -> str:
    ext = pathlib.Path(path).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"

    return "file"

def _call_media_agent(media_type, media_path, prompt):
    if media_type == 'image':
        return media_agent.run(prompt, images=[Image(filepath=media_path)])
    if media_type == 'audio':
        return media_agent.run(prompt, audio=[Audio(filepath=media_path)])
    if media_type == 'video':
        return media_agent.run(prompt, videos=[Video(filepath=media_path)])

    return media_agent.run(prompt, files=[File(filepath=media_path)])


def MediaProcessing(media_path: str = "", question: str = "") -> str:
    """Use this tool to process local media files (image, audio, video, other files).
    Forwards media+question to media_agent in the appropriate format.
    Returns a short single-paragraph or single-sentence description/explaination or/and answer to the question.
    
    Args:
        media_path: local path.
        question: the user's question about the media.
    
    Returns:
        str: description/explaination/answer in a single paragraph/sentence, or 'idk' if not known, or 'file not found' if the file doesn't exist.
    """
    if not media_path or not pathlib.Path(media_path).exists():
        return 'file not found'

    media_type = _detect_media_type(media_path)

    # Build the prompt for media_agent using the convention your media_agent expects:
    # prompt = f"Media: {media_type}={media_path} | Question: {question}"
    prompt = question if question else f"Describe the content of this {media_type} in a single paragraph or sentence."

    # Call media_agent; return its content
    resp = _call_media_agent(media_type, media_path, prompt)
    print(f"MediaProcessing got response: {resp}")
    # resp may be an object with .content; handle that:
    result = getattr(resp, "content", None)
    if result is None:
        # Fallback to string conversion
        return str(resp) or "idk"

    return result

reasoning_agent = Agent(
    model=OpenAILike(
        id="openai/gpt-oss-20b",
        api_key="123",
        base_url="http://192.168.1.157:1234/v1",
        reasoning_effort="high",
        temperature=0.05,
    ),
    tools=[GoogleSearchTools(), MediaProcessing()],
    reasoning=True,
    system_message=(
        "You are a concise assistant."
        "Before answering any question, first think step by step about what tools you can use to find the answer."
        "Use the 'GoogleSearchTools' tool to find relevant information."
        "If the question asks to analyze any media, such as an image, audio or video, or other files, use 'MediaProcessing' tool."
        "Always answer in the shortest possible form:"
        "a single number, a single word, or at most a few words."
        "Never explain or elaborate unless explicitly asked."
        "If you don't know, reply 'idk', do not try to make up an answer. "
    )
)

# reasoning_agent.print_response("Aalyze the image: https://huggingface.co/datasets/hf-internal-testing/fixtures_image/resolve/main/flower.png")

# image_path = Path(__file__).parent.joinpath("temp/image.png")
# reasoning_agent.print_response(
#     "Explain the content of the image in a single sentence. If there's text in the image, include it in your answer."
# )

# print(image_path)
# print(type(image_path))

# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str, media_path: str = None) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        if media_path and Path(media_path).exists():
            media_type = _detect_media_type(media_path)
            # Format the input so the reasoning agent sees the media reference and is likely to call MediaProcessing
            agent_input = f"Media: {media_type}={media_path} | Question: {question}"
        else:
            agent_input = question

        new_answer = reasoning_agent.run(agent_input)
        new_answer_content = new_answer.content
        print(f"Agent returning new answer: {new_answer_content}")
        return new_answer_content
    

agent_1 = BasicAgent()
image_path = Path(__file__).parent.joinpath("temp/image.png")
answer = agent_1("What color is the hair of the character in the image?", media_path=str(image_path))