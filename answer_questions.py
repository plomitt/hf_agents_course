from pathlib import Path
import shutil
import json
import os
from time import time
from agno.tools import tool

from common import get_question_prompt, get_question_prompt_with_media, get_questions_data, is_question_about_media
from reasoning_agent import create_reasoning_agent

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