from pathlib import Path
import shutil
import json
import os
from time import time

from reasoning_agent import detect_media_type, create_reasoning_agent

MEDIA_FILES_DIR = "media_files"
QUESTIONS_FILEPATH = f"{MEDIA_FILES_DIR}/questions.json"
ANSWERS_DIR_PATH = f"{MEDIA_FILES_DIR}/answers"
ANSWERS_PREFIX = "answers_"

def get_questions_data():
    question_data = json.load(open(QUESTIONS_FILEPATH, "r"))
    return question_data

def get_media_file_path(file_id):
    media_files_dir = Path(MEDIA_FILES_DIR)
    matching_files = list(media_files_dir.glob(f'{file_id}.*'))

    if not matching_files:
        print(f"WARNING: No file found with ID '{file_id}' in '{media_files_dir}'")
        return None

    if len(matching_files) > 1:
        print(f"WARNING: Multiple files found for ID '{file_id}': {matching_files}")
        return None

    return matching_files[0]

def is_question_about_media(question_text, file_name):
    return 'https' in question_text or '.' in file_name

def get_question_prompt(task_id, question_text):
    prompt = f"{question_text}\n\ntask_id='{task_id}'"

    return prompt

def get_question_prompt_with_media(task_id, question_text):
    media_path = get_media_file_path(task_id)
    media_type = detect_media_type(str(media_path))

    prompt = f"{question_text}\n\n{media_type}_paths=['{str(media_path)}']\n\ntask_id='{task_id}'"

    return prompt

def get_latest_answers_file():
    answers_files = [f for f in os.listdir(ANSWERS_DIR_PATH) if f.startswith(ANSWERS_PREFIX) and f.endswith(".json")]
    if not answers_files:
        return None
    
    latest_file = max(answers_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

    return latest_file

def get_latest_answers_filepath():
    latest_file = get_latest_answers_file()
    if latest_file is None:
        raise FileNotFoundError("No answers files found.")

    latest_path = os.path.join(ANSWERS_DIR_PATH, latest_file)
    
    return latest_path

def get_latest_answers_index():
    latest_file = get_latest_answers_file()
    if latest_file is None:
        return -1

    latest_index = int(latest_file.split("_")[1].split(".")[0])
    
    return latest_index

def get_latest_answers_data():
    latest_path = get_latest_answers_filepath()
    with open(latest_path, "r") as f:
        answers_data = json.load(f)
    
    return answers_data

def get_new_answers_index():
    latest_index = get_latest_answers_index()
    return latest_index + 1

def get_new_answers_filepath():
    new_answer_id = get_new_answers_index()

    return f"{ANSWERS_DIR_PATH}/{ANSWERS_PREFIX}{new_answer_id}.json"

def create_new_answers_file():
    new_answers_path = get_new_answers_filepath()
    shutil.copy(QUESTIONS_FILEPATH, new_answers_path)

def save_answer_tool(task_id: str, answer: str):
    """Use this tool to save the answer to a specific question by its task_id.
    
    Args:
        task_id: The task_id of the question.
        answer: The answer to the question.
        
    Returns:
        str: Confirmation message that the answer was saved.
    """
    print(f"save_answer_tool received task_id: {task_id}, answer: {answer}")
    try:
        latest_answers_path = get_latest_answers_filepath()
    except FileNotFoundError:
        create_new_answers_file()
        latest_answers_path = get_latest_answers_filepath()

    with open(latest_answers_path, "r") as f:
        answers_data = json.load(f)

    for item in answers_data:
        if item.get("task_id") == task_id:
            item["answer"] = answer
            break
    else:
        answers_data.append({
            "task_id": task_id,
            "question": "",
            "Level": "",
            "file_name": "",
            "answer": answer
        })

    with open(latest_answers_path, "w") as f:
        json.dump(answers_data, f, indent=4)

    output = f"Answer saved for task_id {task_id}"
    print(output)

    return output

if __name__ == "__main__":
    question_data = get_questions_data()
    reasoning_agent = create_reasoning_agent()
    
    for item, counter in enumerate(question_data, start=1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")

        is_about_media = is_question_about_media(question_text, file_name)

        if is_about_media:
            prompt = get_question_prompt_with_media(task_id, question_text)
        else:
            prompt = get_question_prompt(task_id, question_text)
        
        print(f"Processing question {counter}/{len(question_data)}...")
        start_time = time()
        reasoning_agent.run(prompt, stream=False)
        end_time = time()
        print(f"Question {counter}/{len(question_data)} (media={is_about_media}) took {end_time - start_time:.2f} seconds.")