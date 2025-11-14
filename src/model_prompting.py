import ollama

from google import genai


def prompt_model(prompt: str, model: str, client: genai.Client = None) -> str:
    """
    Prompt a language model (local or cloud) and return the response.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The model identifier.
        client: The cloud API client (if using cloud generation).

    Returns:
        str: The response from the model.
    """
    if client is not None:
        return prompt_model_cloud(prompt, model, client)
    return prompt_model_local(prompt, model)
    
def prompt_model_local(prompt: str, model: str) -> str:
    """
    Prompt a local Ollama model and return the response.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The local model identifier.

    Returns:
        str: The response from the model.
    """
    response = ollama.generate(model=model, prompt=prompt)
    return response['response'].strip()

def prompt_model_cloud(prompt: str, model: str, client: genai.Client) -> str:
    """
    Prompt a cloud-based Gemini model and return the response.

    Args:
        prompt (str): The prompt to send to the model.
        model (str): The cloud model identifier.
        client: The cloud API client.

    Returns:
        str: The response from the model.
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text.strip()

# TODO: Delete this part
# import time

# def prompt_model_cloud(prompt: str, model: str, client: genai.Client) -> str:
#     """
#     Prompt a cloud-based Gemini model and return the response.

#     Args:
#         prompt (str): The prompt to send to the model.
#         model (str): The cloud model identifier.
#         client: The cloud API client.

#     Returns:
#         str: The response from the model.
#     """
#     # static storage inside the function object
#     if not hasattr(prompt_model_cloud, "_last_call"):
#         prompt_model_cloud._last_call = 0.0

#     now = time.time()
#     elapsed = now - prompt_model_cloud._last_call

#     wait = 10 - elapsed
#     print(" - Wait: ", wait)
#     if wait > 0:
#         time.sleep(wait)

#     prompt_model_cloud._last_call = time.time()

#     response = client.models.generate_content(
#         model=model,
#         contents=prompt,
#     )
#     return response.text.strip()
