from pathlib import Path
from agno.agent import Agent
from agno.media import Audio

from common import detect_media_type, get_absolute_path, get_model, get_media_path

def create_audio_agent():
    media_agent = Agent(
        model=get_model('open_gemini_flash'),
        instructions="You are a concise audio question-answering assistant. You are given a question and an audio file. You need to answer the question based on the audio file.",
    )
    
    return media_agent

def audio_question_tool(args):
    """Use this tool to ask a specific question about an audio file.
    
    Args:
        audio_path: The exact relative local path to the audio file.
        question: The specific question to ask about the audio.
    
    Returns:
        str: The answer to the question, or 'idk', or 'file not found', or 'file type not supported'.
    """
    audio_path = args.get("audio_path", None)
    question = args.get("question", None)

    print(f"audio_tool received question: {question[:50]}...")

    if not audio_path or not Path(audio_path).exists():
        return 'file not found'
    
    audio_path = get_absolute_path(audio_path)
    print(f"audio_tool received audio_path: {str(audio_path)}")

    media_type = detect_media_type(str(audio_path))

    if media_type == 'audio':
        audio_agent = create_audio_agent()
        answer = audio_agent.run(question, audio=[Audio(filepath=audio_path)])
        print(f"audio_tool received answer: {answer.content}")
        return answer.content
    
    return 'file type not supported'

if __name__ == "__main__":
    media_agent = create_audio_agent()
    media_agent.print_response("What is the audio recording about?",  audio=[Audio(filepath=get_media_path('audio1'))], stream=True, show_tool_calls=True, show_full_reasoning=True)
