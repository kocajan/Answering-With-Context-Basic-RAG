from google import genai

from .model_prompting import prompt_model


def evaluate_context_sufficiency(question: str, context: str, system_prompt: str, model: str, client: genai.Client = None) -> tuple:
    """
    Evaluate if the provided context is sufficient to answer the question.

    Args:
        question (str): The user's question.
        context (str): The retrieved context.
        system_prompt (str): The system prompt for context evaluation.
        model (str): The model identifier.
        client: The cloud API client (if using cloud generation).

    Returns:
        tuple: A tuple containing a boolean indicating sufficiency and a string with missing information if any.
    """
    # Insert question and context into the system prompt
    system_prompt = system_prompt.format(question=question, context=context)

    # Get evaluation response from the model
    evaluation_response = prompt_model(
        prompt=system_prompt,
        model=model,
        client=client
    )

    # On the first line of the response, expect "Sufficient" or "Insufficient"
    first_line = evaluation_response.splitlines()[0].strip().lower()
    is_sufficient = first_line.startswith("sufficient")

    if not is_sufficient and not first_line.startswith("insufficient"):
        print("     - Warning: Unexpected evaluation response format. Assuming insufficient context.")

    # Extract missing information from the rest of the response
    missing_info = "\n".join(evaluation_response.splitlines()[1:]).strip()

    # TODO: Delete debug prints
    # print("==============================")
    # print(system_prompt)
    # print("==============================")
    # print(is_sufficient)
    # print("==============================")
    # print(missing_info)
    # print("==============================")
    # input("Press Enter to continue...")

    return is_sufficient, missing_info
        