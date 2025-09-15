from time import time

from common import SubmitFinalAnswer, create_new_answers_file, get_question_prompt, get_question_prompt_with_media, get_questions_data, is_latest_answers_file_complete, is_question_about_media, is_question_answered
from new_react_agent import create_react_agent, get_final_answer

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

def run_react_agent(task_id, prompt):
    react_agent = create_react_agent()
    answer = react_agent.ask(prompt)
    final = get_final_answer(answer)
    SubmitFinalAnswer(task_id, final)

def run_agent_on_q(task_id, prompt, is_about_media, counter, total):
    print(f"Processing question {counter}/{total}...")
    start_time = time()

    try:
        run_react_agent(task_id, prompt)
    except Exception as e:
        print(f"Agent error on question {counter}/{total}: {e}")

    end_time = time()
    print(f"Question {counter}/{total} (media={is_about_media}) took {end_time - start_time:.2f} seconds.")

def answer_question(item, counter, total):
    task_id = item.get("task_id")
    prompt, is_about_media = get_prompt_info(item)
    run_agent_on_q(task_id, prompt, is_about_media, counter, total)

def answer_all_questions():
    question_data = get_questions_data()
    total_questions = len(question_data)
    
    for counter, item in enumerate(question_data, start=1):
        task_id = item.get("task_id")
        if not is_question_answered(task_id):
            answer_question(item, counter, total_questions)

def keep_answering_until_complete(limit=10):
    for i in range(limit):
        if is_latest_answers_file_complete():
            print("All questions have been answered.")
            break

        answer_all_questions()

if __name__ == "__main__":
    if is_latest_answers_file_complete():
        create_new_answers_file()

    keep_answering_until_complete()