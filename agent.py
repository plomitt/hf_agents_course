import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.models.openrouter import OpenRouter
from agno.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from pydantic import BaseModel, Field
from pprint import pprint

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_model_id = "moonshotai/kimi-k2"


# Using Google AI Studio
# llm_model = Gemini(id="gemini-2.5-flash-lite", temperature=0.01)
# llm_model = OpenRouter(api_key=openrouter_api_key, id=openrouter_model_id)
# llm_model = Ollama(id="gemma3:4b", provider="Ollama")
# llm_model = Ollama(id="llama3.2:1b", provider="Ollama")
llm_model = Ollama(id="qwen3:1.7b", provider="Ollama")

class AgentAnswer(BaseModel):
    rationale: str = Field(..., description="Explaination of the answer.")
    answer: str = Field(..., description="Final answer. Only one word.")


# 1. Web Search Agent
# Role: Retrieve raw search results for a given query.
ai_agent = Agent(
    name="Question Answering Agent",
    role="Specialized in performing web searches and returning raw results.",
    model=llm_model,
    tools=[GoogleSearchTools()],
    instructions=[
        "Use the 'GoogleSearchTools' tool to perform a relevant google search, and use the results to answer the question. Never answer without using the 'GoogleSearchTools' tool.",
    ],
    response_model=AgentAnswer,
    use_json_mode=True,
    show_tool_calls=True,
)

# Example usage
# print("--- Multi-Agent System Test Case 1 ---")
# user_query_1 = "What is the current population of moscow?"
# ai_agent.print_response(user_query_1)
# final_answer_1 = get_concise_answer(user_query_1)
# print(f"\nUser Query: {user_query_1}")
# print(f"Final Answer: '{final_answer_1}'")

# print("\n--- Multi-Agent System Test Case 2 ---")
# user_query_2 = "Who invented the telephone?"
# reasoning_team.print_response(user_query_2)
# final_answer_2 = get_concise_answer(user_query_2)
# print(f"\nUser Query: {user_query_2}")
# print(f"Final Answer: '{final_answer_2}'")

# print("\n--- Multi-Agent System Test Case 3 (Specific Numeric Answer) ---")
# user_query_3 = "How many degrees are in a right angle?"
# final_answer_3 = get_concise_answer(user_query_3)
# print(f"\nUser Query: {user_query_3}")
# print(f"Final Answer: '{final_answer_3}'")

# print("\n--- Multi-Agent System Test Case 4 (Direct Word Answer) ---")
# user_query_4 = "What is the opposite of 'hot'?"
# final_answer_4 = get_concise_answer(user_query_4)
# print(f"\nUser Query: {user_query_4}")
# print(f"Final Answer: '{final_answer_4}'")

# print("\n--- Multi-Agent System Test Case 5 (Question likely needing search, concise answer) ---")
# user_query_5 = "When was the Battle of Hastings fought?"
# final_answer_5 = get_concise_answer(user_query_5)
# print(f"\nUser Query: {user_query_5}")
# print(f"Final Answer: '{final_answer_5}'")

# print("\n--- Multi-Agent System Test Case 6 (No direct answer expected) ---")
# user_query_6 = "What is the meaning of life?"
# final_answer_6 = get_concise_answer(user_query_6)
# print(f"\nUser Query: {user_query_6}")
# print(f"Final Answer: '{final_answer_6}'")



# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        # new_answer = ai_agent.run(question)
        # new_answer_content = new_answer.content.rstrip()
        # print(f"Agent returning new answer: {new_answer_content}")
        return 'new_answer_content'