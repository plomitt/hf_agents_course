import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.team import Team
from agno.tools.googlesearch import GoogleSearchTools
from pprint import pprint


# Using Google AI Studio
llm_model = Gemini(id="gemini-2.0-flash-lite")

# 1. Web Search Agent
# This agent's role is purely to perform searches and return raw results.
web_search_agent = Agent(
    name="Web Search Agent",
    role="Specialized in performing web searches.",
    model=llm_model,
    tools=[GoogleSearchTools()],
    instructions=[
        "Your task is to perform a web search for the exact query provided.",
        "Use the 'Google Search' tool with the given query and return the results directly.",
        "Do NOT summarize, interpret, or add any commentary to the search results.",
        "If no relevant results are found by the tool, simply state 'No relevant search results found by the tool.'",
    ],
    show_tool_calls=True, # Good for debugging the search agent's actions
)

# 2. Answer Composition Agent
# This agent's role is to take information and provide a highly concise, unformatted answer.
answer_composer_agent = Agent(
    name="Answer Composer Agent",
    role="Expert in extracting and composing concise, unformatted answers.",
    model=llm_model,
    instructions=[
        "You are tasked with composing the final answer.",
        "You will receive the original user question and relevant search results/information.",
        "Identify the single most accurate and direct answer to the user's question from the provided information.",
        "Your response MUST contain ONLY the answer. No introductory phrases, explanations, conversational filler, or greetings.",
        "Do NOT use any special formatting (e.g., bolding, italics, bullet points, numbered lists, markdown headers) unless the original question EXPLICITLY asks for it.",
        "If the question asks for a single word, provide only that single word.",
        "If the question asks for a specific piece of data (e.g., a number, a date, a name), provide only that data.",
        "Do NOT explain your reasoning or decision-making process.",
        "Ensure your output is the shortest possible correct answer. Trim any leading/trailing whitespace.",
        "Do NOT reiterate the question or any part of the prompt in your answer.",
        "If you cannot find a direct answer in the provided information, respond with 'Not found.'",
    ],
    show_tool_calls=False, # This agent does not use tools
)

# 3. Reasoning Team (The Team Leader)
# The team leader orchestrates the flow and synthesizes the final output.
reasoning_team = Team(
    name="Reasoning Team",
    mode="coordinate", # Team leader delegates tasks and synthesizes outputs
    model=llm_model,
    members=[web_search_agent, answer_composer_agent],
    instructions=[
        "You are the central coordinator for providing concise answers to user queries.",
        "Your process involves two primary steps, performed sequentially:",
        "1. **Information Gathering:** Analyze the user's original query. Formulate a precise search term based on the query.",
        "   Delegate this search term to the 'Web Search Agent'.",
        "2. **Final Answer Composition:** Once the 'Web Search Agent' returns the search results, combine the original user's query with these results.",
        "   Delegate this combined information to the 'Answer Composer Agent' to formulate the final, direct answer.",
        "**Crucially, your final output MUST be ONLY the concise answer provided by the 'Answer Composer Agent'.**",
        "Do NOT include any planning steps, internal thoughts, delegation messages, or any other extraneous text in your final response.",
        "Your response should be the pure, unformatted answer from the 'Answer Composer Agent'.",
        "Ensure the final output is rigorously trimmed of any leading/trailing whitespace or newlines."
    ],
    # These settings help the team leader effectively coordinate and synthesize
    enable_agentic_context=True, # Allows the team leader to maintain and pass context
    share_member_interactions=True, # Shares member responses with subsequent members/leader
    show_tool_calls=True, # Show calls made by the team leader and its members
    show_members_responses=True, # Show the raw responses from individual members for debugging
    # add_member_tools_to_system_message=False # Might be useful if the leader tries to use member tools directly
)

# Helper function to run the query and apply rstrip for cleanliness
def get_concise_answer(query: str) -> str:
    response = reasoning_team.run(query)
    content = str(response.content) if response.content is not None else ""
    return content.rstrip()

# Example usage
print("--- Multi-Agent System Test Case 1 ---")
user_query_1 = "What is the current population of Tokyo?"
final_answer_1 = get_concise_answer(user_query_1)
print(f"\nUser Query: {user_query_1}")
print(f"Final Answer: '{final_answer_1}'")

print("\n--- Multi-Agent System Test Case 2 ---")
user_query_2 = "Who invented the telephone?"
final_answer_2 = get_concise_answer(user_query_2)
print(f"\nUser Query: {user_query_2}")
print(f"Final Answer: '{final_answer_2}'")

print("\n--- Multi-Agent System Test Case 3 (Specific Numeric Answer) ---")
user_query_3 = "How many degrees are in a right angle?"
final_answer_3 = get_concise_answer(user_query_3)
print(f"\nUser Query: {user_query_3}")
print(f"Final Answer: '{final_answer_3}'")

print("\n--- Multi-Agent System Test Case 4 (Direct Word Answer) ---")
user_query_4 = "What is the opposite of 'hot'?"
final_answer_4 = get_concise_answer(user_query_4)
print(f"\nUser Query: {user_query_4}")
print(f"Final Answer: '{final_answer_4}'")

print("\n--- Multi-Agent System Test Case 5 (Question likely needing search, concise answer) ---")
user_query_5 = "When was the Battle of Hastings fought?"
final_answer_5 = get_concise_answer(user_query_5)
print(f"\nUser Query: {user_query_5}")
print(f"Final Answer: '{final_answer_5}'")

print("\n--- Multi-Agent System Test Case 6 (No direct answer expected) ---")
user_query_6 = "What is the meaning of life?"
final_answer_6 = get_concise_answer(user_query_6)
print(f"\nUser Query: {user_query_6}")
print(f"Final Answer: '{final_answer_6}'")



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