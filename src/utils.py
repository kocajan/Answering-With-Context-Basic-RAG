import os
import yaml
import hashlib


def load_config(config_directory_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_directory_path (str): The path to the configuration directory.

    Returns:
        dict: The loaded configuration.
    """
    # First get all config files in the directory
    full_config = {}
    for file_name in os.listdir(config_directory_path):
        if file_name.endswith(".yaml") or file_name.endswith(".yml"):
            config_path = os.path.join(config_directory_path, file_name)
            
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                full_config.update(config)
    return full_config

def print_answers(answers: dict) -> None:
    """
    Print the answers to the questions.

    Args:
        answers (dict): A dictionary mapping questions to their answers.

    Returns:
        None
    """
    print("\n=== Answers ===\n")
    for question, answer in answers.items():
        print(f" - Question: {question}\n - Answer: \n{answer}\n{'-'*40}\n")
    print("=== End of Answers ===\n")
    
def hash_text(text: str) -> str:
    """
    Generate a SHA-256 hash of the given text.
    
    Args:
        text (str): The input text.

    Returns:
        str: The generated hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()