from google import genai

from .model_prompting import prompt_model


def make_query(parameters: dict, system_prompt: str, model: str, client: genai.Client = None) -> str:
    """
    Generate a query using the specified model.

    Args:
        parameters (dict): A dictionary of parameters that will be formatted into the system prompt.
        system_prompt (str): The system prompt for query generation.
        model (str): The model identifier.
        client: The cloud API client (if using cloud generation).

    Returns:
        str: The generated query.
    """
    prompt = system_prompt.format(**parameters)
    return prompt_model(prompt, model=model, client=client).strip()
