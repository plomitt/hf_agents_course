import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.models.openrouter import OpenRouter
from agno.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from pprint import pprint

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
openrouter_model_id = "moonshotai/kimi-k2"


# Using Google AI Studio
# llm_model = Gemini(id="gemini-2.5-flash-lite")
llm_model = OpenRouter(api_key=openrouter_api_key, id=openrouter_model_id)
# llm_model = Ollama(id="gemma3:4b", provider="Ollama")
# llm_model = Ollama(id="llama3.2:1b", provider="Ollama")

# 1. Web Search Agent
# Role: Retrieve raw search results for a given query.
web_search_agent = Agent(
    name="Web Search Agent",
    role="Specialized in performing web searches and returning raw results.",
    model=llm_model,
    tools=[GoogleSearchTools()],
    instructions=[
        "Your sole task is to perform a web search for the exact query provided.",
        "Use the 'Google Search' tool and return the search results directly.",
        "Do NOT summarize, interpret, or add any commentary to the search results.",
        "If no relevant results are found by the tool, simply state 'No relevant search results found by the tool.'",
    ],
    show_tool_calls=True, # Keep for debugging the search process
)

# 2. Reasoning Team (The Coordinator / Final Condenser)
# Role: Orchestrates the process, then condenses the detailed answer into the final, concise output.
reasoning_team = Team(
    name="Reasoning Team",
    mode="coordinate", # Team leader delegates tasks and synthesizes/condenses outputs
    model=llm_model,
    members=[web_search_agent],
    description="You are an advanced AI coordinator tasked with answering user questions precisely and concisely.",
    instructions=[
        "You will receive an original user question and a set of search results.",
        "Your task is to find the most accurate and concise answer possible.",
        "Follow these steps:",
        "1. **Search:** First, delegate the user's original query to the 'Web Search Agent' to retrieve relevant information.",
        "2. **Condense and Finalize:** Once you receive the answer from the 'Web Search Agent', your final and most important task is to **condense that detailed answer into the most concise, direct, and unformatted answer possible that directly addresses the user's original question.**",
        "Your output MUST contain ONLY this condensed answer. Do NOT include any planning steps, internal thoughts, explanations, delegation messages, or any other extraneous text.",
        "Do NOT use any special formatting (e.g., bolding, italics, bullet points, markdown) unless the original user question EXPLICITLY asks for it.",
        "If the original question asks for a single word or number, provide only that word or number.",
        "The answer MUST be a single word, a single number, or a very short phrase (e.g., a person's name or a date).",
        "The answer MUST NOT be a paragraph or even a sentence, unless specified by the question.",
        "Do not include any other text, explanations, or commentary.",
        "If the answer is a number, only provide the number. Write it without anny commas or full stops, or special characters for currency, %, etc.",
        "If the answer is a string, do not use abbreviations or acronyms.",
        "If the answer is a list, make sure there is only one item per comma, and one space after the comma.",
        "Examples:",
        "Question: 'What is the capital of France?' -> Answer: 'Paris'",
        "Question: 'What is the population of Beijing?' -> Answer: '20000000'",
        "Question: 'How many degrees are in a right angle?' -> Answer: '90'",
        "Question: 'Who invented the light bulb?' -> Answer: 'Thomas Edison'",
        "Question: 'Name three planets from the solar system.' -> Answer: 'Earth, Mars, Venus'",
        "If, after all steps, a direct and accurate answer cannot be found, simply respond with 'Not found.'",
    ],
    enable_agentic_context=True,      # Allows the team leader to maintain and pass context between steps
    share_member_interactions=True,   # Shares all member responses with subsequent members/leader
    show_tool_calls=True,             # Show calls made by the team leader and its members
    show_members_responses=True,      # Show the raw responses from individual members for debugging
    # add_member_tools_to_system_message=False # Can sometimes help the leader delegate more cleanly
)

# Helper function to run the query and apply rstrip for cleanliness
def get_concise_answer(query: str) -> str:
    response = reasoning_team.run(query)
    content = str(response.content) if response.content is not None else ""
    return content.rstrip()

# Example usage
print("--- Multi-Agent System Test Case 1 ---")
user_query_1 = "What is the current population of Tokyo?"
reasoning_team.print_response(user_query_1)
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

        new_answer = reasoning_team.run(question)
        new_answer_content = new_answer.content.rstrip()
        print(f"Agent returning new answer: {new_answer_content}")
        return new_answer_content