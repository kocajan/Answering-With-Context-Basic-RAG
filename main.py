from src.utils import load_config, print_answers
from src.question_answering import get_answers_for_questions


def main() -> None:
    """
    Entry point for the module.

    Returns:
        None
    """
    # Load configuration
    config_directory_path = "cfg/"
    config = load_config(config_directory_path)

    # Define questions
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What are the benefits of regular exercise?"
    ]

    # Get answers for questions
    print("Starting question answering process...")
    answers = get_answers_for_questions(config, questions)

    # Print the answers
    print("Printing answers...")
    print_answers(answers)

if __name__ == "__main__":
    main()
