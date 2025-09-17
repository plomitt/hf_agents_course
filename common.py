from typing import Any, Dict, List
from urllib.parse import urlparse
from agno.models.openai.like import OpenAILike
from agno.models.openrouter import OpenRouter
from agno.models.google import Gemini
from dotenv import load_dotenv
from pathlib import Path
import shutil
import json
import os
import ast
import operator as op

import pandas as pd

from simple_agent import ToolSpec

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

BASE_REACT_TASK_PROMPT = """The final answer should adhere to the following rules: always answer in the shortest possible form: a single number, a single word, or at most a few words; never explain or elaborate unless explicitly asked; omit punctuation at the end of sentences/words, and omit any units unless explicitly asked; if the answer is a list of items, it should be a comma separated list. If you don't know the answer, or can't process a file type, the final answer should be simply 'idk'."""

def get_question_prompt(task_id, question_text):
    prompt = f"{question_text}\n\n{BASE_REACT_TASK_PROMPT}"

    return prompt

def get_question_prompt_with_media(task_id, question_text):
    media_path = get_media_file_path(task_id)
    media_type = detect_media_type(str(media_path))

    prompt = f"{question_text}\n\n{media_type}_paths=['{str(media_path)}']\n\n{BASE_REACT_TASK_PROMPT}\n\nUse the tools at your disposal to answer the question based on the {media_type} file."

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
    return answer is not None and answer != "" and answer != "idk"

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

def get_absolute_path(media_path: str) -> Path:
    try:
        return Path(__file__).parent.joinpath(media_path)
    except Exception as e:
        error = f"Error getting absolute path: {e}"
        print(error)
        return error
    
def csv_to_text(args) -> str:
    """Convert a CSV file to a text representation.

    Args:
        file_path: The path to the CSV file.

    Returns:
        str: The text representation of the CSV file.
    """
    file_path = args.get("file_path", None)
    if not file_path:
        return "No file path provided"

    try:
        path = get_absolute_path(file_path)
        print(f"csv_to_text reading file: {str(path)}")
        df = pd.read_csv(path)
        return df.to_string()
    except Exception as e:
        return f"Error reading CSV file: {e}"

def code_to_text(args) -> str:
    """Convert a code file to a text representation.

    Args:
        file_path: The path to the code file.

    Returns:
        str: The text representation of the code file.
    """
    file_path = args.get("file_path", None)
    if not file_path:
        return "No file path provided"

    try:
        path = get_absolute_path(file_path)
        print(f"code_to_text reading file: {str(path)}")
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading code file: {e}"

def _split_transcript_and_final(text: str) -> tuple[str, str]:
    """Return (transcript, final_answer_text) using the last 'Final Answer:' marker."""
    if not isinstance(text, str) or not text:
        return "", ""
    i = text.rfind("Final Answer:")
    if i < 0:
        return text, ""
    return text[:i].rstrip(), text[i + len("Final Answer:"):].strip()

def get_final_answer(answer):
    transcript, final = _split_transcript_and_final(answer.content)
    return final

def make_web_search_tool(default_region: str, default_time: str | None, default_max: int) -> ToolSpec:
    """DuckDuckGo search (via ddgs). Returns compact results."""
    SERP_HOSTS = {
        "bing.com", "www.bing.com",
        "google.com", "www.google.com",
        "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
        "startpage.com", "www.startpage.com",
        "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
        "baidu.com", "www.baidu.com",
    }

    def _is_serp(url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if not host:
                return False
            if host in SERP_HOSTS:
                return True
            # Common SERP path patterns
            if any(seg in (p.path or "").lower() for seg in ("/search", "/html", "/lite")) and (
                host.endswith("google.com") or host.endswith("bing.com") or host.endswith("duckduckgo.com") or host.endswith("yahoo.com")
            ):
                return True
            return False
        except Exception:
            return False

    def handler(args: Dict[str, Any]) -> Any:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "empty_query"}
        
        print(f"web_search_tool received query: {query}")

        # Use the new ddgs package exclusively
        try:
            from ddgs import DDGS  # type: ignore
        except Exception as e:
            return {"error": f"ddgs_import_failed: {e}"}

        region = str(args.get("region", default_region) or default_region)
        timelimit = args.get("time", default_time)
        max_results = int(args.get("max_results", default_max) or default_max)
        max_results = max(3, min(max_results, 15))

        out: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddg:
                for r in ddg.text(query, region=region, safesearch="moderate", timelimit=timelimit, max_results=max_results):
                    url = r.get("href")
                    if not url or _is_serp(url):
                        # Skip search engine result pages; they are not sources
                        continue
                    out.append({
                        "title": r.get("title"),
                        "url": url,
                        "snippet": r.get("body"),
                    })
        except Exception as e:
            return {"error": f"ddgs_search_failed: {e}", "query": query}

        # If filtering removed everything, still return empty list (let model refine query)
        return {"query": query, "results": out[:max_results]}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "minimum": 3, "maximum": 15, "default": default_max},
            "region": {"type": "string", "default": default_region},
            "time": {"type": ["string", "null"], "enum": [None, "d", "w", "m", "y"], "default": default_time},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return ToolSpec(name="web_search", description="Search the web via ddgs (DuckDuckGo) and return top results (title, url, snippet)", parameters=params, handler=handler)

def make_fetch_page_tool(default_max_chars: int) -> ToolSpec:
    """HTTP GET a URL and extract readable text using BeautifulSoup."""
    UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

    SERP_HOSTS = {
        "bing.com", "www.bing.com",
        "google.com", "www.google.com",
        "duckduckgo.com", "www.duckduckgo.com",
        "search.yahoo.com", "yahoo.com", "www.yahoo.com",
        "startpage.com", "www.startpage.com",
        "yandex.com", "www.yandex.com", "yandex.ru", "www.yandex.ru",
        "baidu.com", "www.baidu.com",
    }

    def _is_serp(url: str) -> bool:
        try:
            p = urlparse(url)
            host = (p.hostname or "").lower()
            if not host:
                return False
            if host in SERP_HOSTS:
                return True
            if any(seg in (p.path or "").lower() for seg in ("/search", "/html", "/lite")) and (
                host.endswith("google.com") or host.endswith("bing.com") or host.endswith("duckduckgo.com") or host.endswith("yahoo.com")
            ):
                return True
            return False
        except Exception:
            return False

    def _extract_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # local import
        except Exception:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        parts: List[str] = []
        # Prefer headings and paragraphs to keep it compact
        for sel in ["h1", "h2", "h3", "p", "li"]:
            for el in soup.select(sel):
                text = (el.get_text(" ", strip=True) or "").strip()
                if text:
                    parts.append(text)
        text = "\n".join(parts)
        # Collapse whitespace
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def handler(args: Dict[str, Any]) -> Any:
        import requests
        url = str(args.get("url", "")).strip()
        if not url:
            return {"error": "empty_url"}
        
        print(f"fetch_page_tool received url: {url}")

        # Block fetching search engine result pages to avoid garbage content
        if _is_serp(url):
            return {"error": "blocked_serp_url", "url": url}
        max_chars = int(args.get("max_chars", default_max_chars) or default_max_chars)
        max_chars = max(1000, min(max_chars, 15000))
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": UA})
            resp.raise_for_status()
        except Exception as e:
            return {"error": f"fetch_failed: {e}", "url": url}

        title = None
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except Exception:
            title = None

        text = _extract_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars]
        return {"url": url, "title": title, "text": text, "length": len(text)}

    params: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Page URL to fetch"},
            "max_chars": {"type": "integer", "minimum": 1000, "maximum": 15000, "default": default_max_chars},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    return ToolSpec(name="fetch_page", description="Fetch a web page and return cleaned text (capped by max_chars)", parameters=params, handler=handler)

def make_calc_tool() -> ToolSpec:
    """Create a safe calculator tool using a tiny AST evaluator."""
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")

    def handler(args):
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return {"error": "empty_expression"}
        try:
            tree = ast.parse(expr, mode="eval")
            val = _eval(tree.body)  # type: ignore[arg-type]
            return {"expression": expr, "value": val}
        except Exception as e:
            return {"error": str(e)}

    params = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
        "required": ["expression"],
        "additionalProperties": False,
    }
    return ToolSpec(name="calc", description="Evaluate basic arithmetic expression and return a JSON result", parameters=params, handler=handler)