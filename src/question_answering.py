from google import genai

from .query_generation import make_query
from .text_summarization import summarize_pages
from .answering_with_context import answer_with_context
from .context_evaluation import evaluate_context_sufficiency
from .google_page_extraction import google_search, extract_from_urls
from .indexing import retrieve_context_from_index, save_context_to_index


def get_answers_for_questions(config: dict, questions: list) -> dict:
    """
    Get answers for a list of questions using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
        questions (list): A list of questions.

    Returns:
        dict: A dictionary mapping questions to their answers.
    """
    # Get the client if using cloud
    if config['use_cloud']:
        client = genai.Client(api_key=config['api_keys']['gemini_api_key'])
    else:
        client = None

    # Get version
    version = "cloud" if client else "local"

    # Get model
    model = config['model'][version]

    # Iterate over questions and get answers
    answers = {}
    print("Processing questions on the", "cloud..." if client else "local machine...")
    for question in questions:
        print(f"- Processing question: {question}")
        answers[question] = get_answer_for_question(config, question, model, version, client)

    return answers

def get_answer_for_question(config: dict, question: str, model: str, version: str, client: genai.Client = None) -> str:
    """
    Get an answer for a single question using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
        question (str): The user's question.
        model (str): The model identifier.
        version (str): The version type ("local" or "cloud").
        client: The cloud API client (if using cloud generation).

    Returns:
        str: The answer to the question.
    """
    # Generate index query
    print("     - Generating index query...")
    index_query_parameters = {
        "question": question
    }
    index_query = make_query(
        parameters=index_query_parameters,
        system_prompt=config["system_prompts"]["index_query_generation"][version],
        model=model,
        client=client
    )

    # Try to retrieve context for the question
    print("     - Retrieving context from index...")
    context = retrieve_context_from_index(index_query,
                                          database_path=config['index']['database_path'],
                                          collection_name=config['index']['collection_name'],
                                          embedding_model_identifier=config['index']['embedding_model_identifier'],
                                          top_k=config['context_retrieval']['top_k'])

    # Iterate until sufficient context is found or max attempts reached
    max_attempts = config["context_retrieval"]["max_attempts"]
    search_query = "" # - NOTE: Could be also initialized to index_query?
    urls = []
    for attempt in range(max_attempts):
        print("     - Attempt", attempt + 1)
        # Evaluate context sufficiency
        print("     - Evaluating context sufficiency...")
        is_sufficient_context, missing_info = \
            evaluate_context_sufficiency(
                question=question,
                context=context,
                system_prompt=config['system_prompts']['context_evaluation'][version],
                model=model,
                client=client
            )

        # If sufficient context, break the loop
        if is_sufficient_context or attempt == max_attempts - 1:
            print(f"     - Finishing context retrieval with {'sufficient' if is_sufficient_context else 'insufficient'} context.")
            break
        print("     - The context is insufficient. Gathering more information...")

        # Generate a more detailed search query
        print("     - Generating search query...")
        search_query_parameters = {
            "question": question,
            "previous_query": search_query,
            "missing_info": missing_info
        }
        search_query = make_query(
            parameters=search_query_parameters,
            system_prompt=config['system_prompts']['search_query_generation'][version],
            model=model,
            client=client
        )

        # Perform Google search
        print("     - Performing Google search...")
        urls = google_search(
            search_query,
            api_key=config['api_keys']['google_search_api_key'],
            cse_id=config['custom_search_engine_id'],
            n=config['search_results']['num_results'],
            timeout=config['search_results']['timeout']
        )

        # Fetch and extract text from URLs
        print("     - Fetching and extracting text from URLs...")
        pages = extract_from_urls(urls)

        # Summarize pages
        print("     - Summarizing extracted texts into new context...")
        context, summaries = summarize_pages(
            pages=pages,
            system_prompt=config['system_prompts']['summarization'][version],
            model=model,
            client=client,
            text_length_limit=config['summarization']['text_length_limit']
        )

    # Check if context is sufficient after exiting loop
    if not is_sufficient_context:
        print("     - Warning: Proceeding with insufficient context after maximum attempts.")

    # Save context to index
    if is_sufficient_context and attempt > 0:
        print("     - Saving context to index...")
        save_context_to_index(
            context_with_sources=summaries,
            database_path=config['index']['database_path'],
            collection_name=config['index']['collection_name'],
            embedding_model_identifier=config['index']['embedding_model_identifier']
        )

    # Answer the question
    print("     - Generating final answer...")
    answer = answer_with_context(
        question=question,
        context=context,
        system_prompt=config['system_prompts']['answer_generation'][version],
        model=model,
        client=client
    )

    return answer
