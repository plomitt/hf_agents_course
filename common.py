from agno.models.openai.like import OpenAILike
from agno.models.openrouter import OpenRouter
from agno.models.google import Gemini
from dotenv import load_dotenv
from pathlib import Path
import shutil
import json
import os

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

MEDIA_FILES_DIR = "media_files"
QUESTIONS_FILEPATH = f"{MEDIA_FILES_DIR}/questions.json"
ANSWERS_DIR_PATH = f"{MEDIA_FILES_DIR}/answers"
ANSWERS_PREFIX = "answers_"

def generate_openrouter_model(model_id='google/gemini-2.5-flash-lite', temperature=0.0):
    return OpenRouter(
            api_key=openrouter_api_key,
            id=model_id,
            temperature=temperature,
        )

def generate_gemini_model(model_id='gemini-2.5-flash-lite', temperature=0.0):
    return Gemini(id=model_id, temperature=temperature)

def generate_local_model(model_id='openai/gpt-oss-20b', temperature=0.0):
    return OpenAILike(
        id=model_id,
        api_key="123",
        base_url="http://192.168.1.157:1234/v1",
        reasoning_effort="high",
        temperature=temperature,
    )

MODELS = {
    'google_gemini_flash': generate_gemini_model('gemini-2.5-flash'),
    'google_gemini_flash_lite': generate_gemini_model('gemini-2.5-flash-lite'),
    'open_gemini_flash': generate_openrouter_model('google/gemini-2.5-flash'),
    'open_gemini_flash_lite': generate_openrouter_model('google/gemini-2.5-flash-lite'),
    'open_dusk_alpha': generate_openrouter_model('openrouter/sonoma-dusk-alpha'),
    'local_oss': generate_local_model('openai/gpt-oss-20b'),
}

def get_model(id):
    try:
        return MODELS[id]
    except KeyError:
        raise ValueError(f"Model with id '{id}' not found.")

MEDIA_FILE_PATHS = {
    'audio1': 'media_files/1f975693-876d-457b-a649-393859e79bf3.mp3',
    'table1': 'media_files/7bd855d8-463d-4ed5-93ca-5fe35145f733.csv',
    'video1': 'media_files/9d191bce-651d-4746-be2d-7ef8ecadb9c2.mp4',
    'audio2': 'media_files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3',
    'video2': 'media_files/a1e91b78-d3d8-4675-bb8d-62741b4b68a6.mp4',
    'image1': 'media_files/cca530fc-4052-43b2-b130-b30968d8aa44.png',
    'code1': 'media_files/f918266a-b3e0-4914-865d-4faa564f1aef.py'
}

def get_media_path(id):
    try:
        return MEDIA_FILE_PATHS[id]
    except KeyError:
        raise ValueError(f"Media file with id '{id}' not found.")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
CODE_EXTS = {".py", ".js", ".json", ".html", ".css", ".java", ".c", ".cpp", ".cs", ".rb", ".go", ".rs", ".ts"}
SPREADSHEET_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}
TEXT_EXTS = {".txt", ".md", ".pdf", ".docx"}

def detect_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
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

def get_latest_answers_last_answered_q_number():
    answers_data = get_latest_answers_data()
    answered_q_numbers = [item.get("q_number", 0) for item in answers_data if item.get("answer")]

    if not answered_q_numbers:
        return 0

    return max(answered_q_numbers)

def is_question_item_answered(question_item):
    answer = question_item.get("answer", None)
    return answer is not None and answer != ""

def is_question_answered(task_id):
    answers_data = get_latest_answers_data()
    for item in answers_data:
        if item.get("task_id") == task_id:
            return is_question_item_answered(item)
    return False

def is_latest_answers_file_complete():
    answers_data = get_latest_answers_data()
    for item in answers_data:
        if not is_question_item_answered(item):
            return False
    return True

def get_new_answers_index():
    latest_index = get_latest_answers_index()
    return latest_index + 1

def get_new_answers_filepath():
    new_answer_id = get_new_answers_index()

    return f"{ANSWERS_DIR_PATH}/{ANSWERS_PREFIX}{new_answer_id}.json"

def create_new_answers_file():
    new_answers_path = get_new_answers_filepath()
    shutil.copy(QUESTIONS_FILEPATH, new_answers_path)

def SubmitFinalAnswer(task_id: str, answer: str):
    """Use this tool to submit the final answer to the user's question by task_id.
    
    Args:
        task_id: The task_id of the question.
        answer: The synthesized, final answer.
        
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