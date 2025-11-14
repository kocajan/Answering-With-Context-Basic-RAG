from google import genai

from .model_prompting import prompt_model


def summarize_text_local(text: str, system_prompt: str, model: str) -> str:
    """
    Summarize text using the local Ollama model.

    Args:
        text (str): The text to summarize.
        system_prompt (str): The system prompt for summarization.
        model (str): The local model identifier.

    Returns:
        str: The summarized text.
    """
    prompt = system_prompt.format(text=text)
    response = prompt_model(prompt, model)
    return response.strip()

def summarize_text_cloud(text: str, system_prompt: str, model:str, client: genai.Client) -> str:
    """
    Summarize text using the cloud-based model.

    Args:
        text (str): The text to summarize.
        system_prompt (str): The system prompt for summarization.
        model (str): The cloud model identifier.
        client: The cloud API client.

    Returns:
        str: The summarized text.
    """
    prompt = system_prompt.format(text=text)
    response = prompt_model(prompt, model=model, client=client)
    return response.strip()


def summarize_pages(pages: dict, system_prompt: str, model: str, client: genai.Client = None, text_length_limit: int = 5000) -> tuple:
    """
    Summarize multiple pages of text locally or using the cloud if a client is provided.
    The result should be a textual context for answering questions and a dictionary of individual summaries.

    Args:
        pages (dict): A dictionary mapping URLs to their extracted text.
        system_prompt (str): The system prompt for summarization.
        model (str): The model identifier.
        client: The cloud API client (if using cloud summarization).
        text_length_limit (int): The maximum length of text to summarize.

    Returns:
        tuple: A tuple containing the concatenated context string and a dictionary of summaries.
    """
    summaries = {}
    context = ""
    for url, text in pages.items():
        short_text = text[:text_length_limit]
        if client is not None:
            summary = summarize_text_cloud(short_text, system_prompt, model, client)
        else:
            summary = summarize_text_local(short_text, system_prompt, model)
        summaries[url] = summary
        context += summary + "\n\n"
    return context.strip(), summaries
