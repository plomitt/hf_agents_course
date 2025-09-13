from time import time

from common import create_new_answers_file, get_latest_answers_last_answered_q_number, get_question_prompt, get_question_prompt_with_media, get_questions_data, is_latest_answers_file_complete, is_question_about_media
from reasoning_agent import create_reasoning_agent

def get_item_info(item):
    task_id = item.get("task_id")
    question_text = item.get("question")
    file_name = item.get("file_name")

    return task_id, question_text, file_name

def get_prompt(is_about_media, task_id, question_text):
    if is_about_media:
        return get_question_prompt_with_media(task_id, question_text)
    else:
        return get_question_prompt(task_id, question_text)

def get_prompt_info(item):
    task_id, question_text, file_name = get_item_info(item)

    is_about_media = is_question_about_media(question_text, file_name)

    prompt = get_prompt(is_about_media, task_id, question_text)

    return prompt, is_about_media

def run_agent_on_q(agent, prompt, is_about_media, counter, total):
    print(f"Processing question {counter}/{total}...")
    start_time = time()
    agent.run(prompt, stream=False)
    end_time = time()
    print(f"Question {counter}/{total} (media={is_about_media}) took {end_time - start_time:.2f} seconds.")

def answer_question(item, counter, total, agent):
    prompt, is_about_media = get_prompt_info(item)
    run_agent_on_q(agent, prompt, is_about_media, counter, total)

def answer_all_questions_from_n(n=1):
    question_data = get_questions_data()
    total_questions = len(question_data)
    reasoning_agent = create_reasoning_agent()
    
    for counter, item in enumerate(question_data, start=1):
        if item.get("q_number", 0) >= n:
            answer_question(item, counter, total_questions, reasoning_agent)

def answer_all_questions_from_start():
    answer_all_questions_from_n(1)

def continue_answering_all_questions():
    last_q_number = get_latest_answers_last_answered_q_number()
    start_q_number = last_q_number + 1
    answer_all_questions_from_n(start_q_number)

if __name__ == "__main__":
    if is_latest_answers_file_complete():
        create_new_answers_file()
        answer_all_questions_from_start()
    else:
        continue_answering_all_questions()