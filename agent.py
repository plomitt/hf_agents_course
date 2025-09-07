import os
from os import getenv
from pathlib import Path
from agno.agent import Agent, RunResponse
from agno.models.openai.like import OpenAILike
from agno.tools.googlesearch import GoogleSearchTools
from agno.models.openrouter import OpenRouter
from agno.media import Image, Audio, Video
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_model_id = "google/gemini-2.0-flash-lite-001"

reasoning_agent = Agent(
    model=OpenAILike(
        id="openai/gpt-oss-20b",
        api_key="123",
        base_url="http://192.168.1.157:1234/v1",
        reasoning_effort="high",
        temperature=0.05,
    ),
    tools=[GoogleSearchTools()],
    reasoning=True,
    system_message=(
        "You are a concise assistant."
        "Before answering any question, first think step by step about what tools you can use to find the answer."
        "Use the 'GoogleSearchTools' tool to find relevant information."
        "If the question asks to analyze an image, audio or video, just say you can't process that type of input, skip reasoning and tool use and directly say you can't process that type of input."
        "Always answer in the shortest possible form:"
        "a single number, a single word, or at most a few words."
        "Never explain or elaborate unless explicitly asked."
        "If you don't know, reply 'idk', do not try to make up an answer. "
    )
)

media_agent = Agent(
    model=OpenRouter(
        api_key=openrouter_api_key,
        id=openrouter_model_id,
        temperature=0.05,
    ),
    tools=[
        Image(),
        Video(),
        Audio(),
    ],
    reasoning=True,
    system_message=(
        "You are a concise media assistant. "
        "Before answering, think step-by-step about which tool(s) to use. "
        "Use Image/Video/Audio tools to process media inputs. "
        "When the user provides or references media, call the appropriate media tool and return its result. "
        "Explain what the media is about. "
        "Also include any text content or important information/facts that are present in the media. "
        "Always limit the final answer to a single sentence. "
        "If you don't know, reply 'idk', do not try to make up an answer. "
    )
)


# reasoning_agent.print_response("Aalyze the image: https://huggingface.co/datasets/hf-internal-testing/fixtures_image/resolve/main/flower.png")

image_path = Path(__file__).parent.joinpath("temp/image.png")
reasoning_agent.print_response(
    "Explain the content of the image in a single sentence. If there's text in the image, include it in your answer.",
    images=[Image(filepath=image_path)],
)

# --- Basic Agent Definition ---
# class BasicAgent:
#     def __init__(self):
#         print("BasicAgent initialized.")
#     def __call__(self, question: str) -> str:
#         print(f"Agent received question (first 50 chars): {question[:50]}...")

#         new_answer = reasoning_agent.run(question)
#         new_answer_content = new_answer.content
#         print(f"Agent returning new answer: {new_answer_content}")
#         return new_answer_content