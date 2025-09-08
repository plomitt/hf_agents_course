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
from agno.tools import tool
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
# openrouter_model_id = "google/gemini-2.5-flash-lite"
openrouter_model_id = "openrouter/sonoma-sky-alpha"

media_agent = Agent(
    model=OpenRouter(
        api_key=openrouter_api_key,
        id=openrouter_model_id,
        temperature=0.05,
    ),
    reasoning=True,
    system_message=(
        "You are a concise media description assistant. "
        "When the user provides or references media in his question, analyze the media and return its description/explaination that will be helpful to answer the question. "
        "Explain what the media is about. "
        "Also include any text content or important/relevant information/facts that are present in the media. "
        "If the media is an image, describe its visual content. "
        "If the media is an audio, transcribe its spoken content and summarize any important sounds. "
        "If the media is a video, summarize its visual and audio content. "
        "If the media is a code file, rewrite it in plain text. "
        "If the media is a spreadsheet, rewrite its tabular content in plain text, using ASCII characters. "
        "If the media is a text file, summarize its main points. "
        "Do not answer the question yet, just describe/explain the media content. "
        "Do include any information/facts that are present in the media that may be useful to answer the question. "
        "Always limit the final answer to a reasonable length. "
        "If you don't know, reply 'idk', do not try to make up an answer. "
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
    files=[File(filepath=media_path)] if media_type not in {'image', 'audio', 'video'} else None


    # print(f"Calling media_agent with prompt: {prompt}")
    # print(f"Calling media_agent with media_type: {media_type}")
    # print(f"with images: {images}")
    # print(f"with audio: {audio}")
    # print(f"with videos: {videos}")
    # print(f"with files: {files}")

    return media_agent.run(prompt, images=images, audio=audio, videos=videos, files=files)


def MediaProcessingTool(media_path: str = "", question: str = "") -> str:
    """Use this tool to process local media files (image, audio, video, code, spreadsheet, text).
    
    Args:
        media_path: local path.
        question: the user's question about the media.
    
    Returns:
        str: description/explaination/answer in a single paragraph/sentence, or 'idk' if not known, or 'file not found' if the file doesn't exist.
    """

    if not media_path or not pathlib.Path(media_path).exists():
        return 'file not found'
    
    media_path = str(media_path)
    print(f"MediaProcessing received media_path: {media_path}")

    media_type = _detect_media_type(media_path)

    # Build the prompt for media_agent using the convention your media_agent expects:
    prompt = f"Describe the content of this {media_type}, including information relevant to the following user question. " + question

    # Call media_agent; return its content
    resp = _call_media_agent(media_type, media_path, prompt)
    result = resp.content
    print(f"MediaProcessing got response: {result}")

    return result

reasoning_agent = Agent(
    model=OpenAILike(
        id="openai/gpt-oss-20b",
        api_key="123",
        base_url="http://192.168.1.157:1234/v1",
        reasoning_effort="high",
        temperature=0.05,
    ),
    tools=[GoogleSearchTools(), MediaProcessingTool],
    reasoning=True,
    system_message=(
        "You are a concise assistant that uses tools to answer questions."
        "Before answering any question, first think deeply and step by step about what tools you should use to find the answer."
        "Use the 'GoogleSearchTools' tool to find relevant information."
        "ALWAYS USE 'GoogleSearchTools' when the user asks about current events, recent news, or anything that may have changed in the last 2 years."
        "Use the 'MediaProcessingTool' tool to get information about any media files: such as an image, audio or video."
        "ALWAYS USE 'MediaProcessingTool' when the user provides or references an image, audio, video, code, or spreadsheet files in his question."
        "IMPORTANT: currrently video cannot be processed due to system limitations, so if the user provides a video, reply 'can't process video'."
        "After calling the tools, reason step by step about the information you get from the tools, and how it relates to the question, then formulate your final answer."
        "Do not just copy and paste the content from the tools; instead, synthesize the information to provide a concise and accurate answer."
        "Reason step by step about the information you get from the tools, and how it relates to the question, then formulate your final answer."
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
            # Format the input so the reasoning agent sees the media reference and is likely to call MediaProcessing
            agent_input = f"Media: {media_type}={media_path} | Question: {question}"
        else:
            agent_input = question

        new_answer_content = _handle_paused_run(reasoning_agent, agent_input)

        return new_answer_content


# --- Example Calls ---

# agent_1 = BasicAgent()
# media_path = Path(__file__).parent.joinpath("media_files/7bd855d8-463d-4ed5-93ca-5fe35145f733.csv")
# answer = agent_1("The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places.", media_path=str(media_path))
# print(answer)

# reasoning_agent.print_response("Media: image=/Volumes/Crucial/programming/hf_agents_course/fa1/temp/image.png | Question: What color is the hair of the character in the image?")

# image_path = Path(__file__).parent.joinpath("temp/image.png")
# media_agent.print_response(
#     "What color is the hair of the character in the image?",
#     images=[Image(filepath=image_path)]
# )

# image_path = Path(__file__).parent.joinpath("temp/image1.png")
# answer = MediaProcessingTool(str(image_path), "What color is the hair of the character in the image?")
# print(answer)