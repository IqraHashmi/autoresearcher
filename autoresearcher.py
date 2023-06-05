#!/usr/bin/env python3
import os
import sys
import json
from termcolor import colored
from dotenv import load_dotenv
import openai

from symantic_scholar_loader import SemanticScholarLoader

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


def openai_call(
        prompt: str, use_gpt4: bool = False, temperature: float = 0.5, max_tokens: int = 100
):
    """
    Calls OpenAI API to generate a response to a given prompt.
    Args:
      prompt (str): The prompt to generate a response to.
      use_gpt4 (bool, optional): Whether to use GPT-4 or GPT-3.5. Defaults to False.
      temperature (float, optional): The temperature of the response. Defaults to 0.5.
      max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.
    Returns:
      str: The generated response.
    Examples:
      >>> openai_call("Hello, how are you?")
      "I'm doing great, thanks for asking!"
    Notes:
      The OpenAI API key must be set in the environment variable OPENAI_API_KEY.
    """
    if not use_gpt4:
        # Call GPT-3.5 turbo model
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content.strip()
    else:
        # Call GPT-4 chat model
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        return response.choices[0].message.content.strip()


# Generate keyword combinations for a given research question
def generate_keyword_combinations(research_question):
    """
    Generates keyword combinations for a given research question.
    Args:
      research_question (str): The research question to generate keyword combinations for.
    Returns:
      list: A list of keyword combinations for the given research question.
    Examples:
      >>> generate_keyword_combinations("What is the impact of AI on healthcare?")
      ["AI healthcare", "impact AI healthcare", "AI healthcare impact"]
    """
    keyword_combination_prompt = """
    `reset`
    `no quotes`
    `no explanations`
    `no prompt`
    `no self-reference`
    `no apologies`
    `no filler`
    `just answer`

    Generate several keyword combinations based on the following research question: {research_question}.
    Don't generate more than 5 keyword combinations.

    The output should be structured like this:
    Write "KeywordCombination:" and then list the keywords like so "Keyword,Keyword,Keyword"

    """
    prompt = keyword_combination_prompt.format(research_question=research_question)
    response = openai_call(prompt, use_gpt4=False, temperature=0, max_tokens=200)
    combinations = response.split("\n")
    return [
        combination.split(": ")[1]
        for combination in combinations
        if ": " in combination
    ]


def literature_review(research_question, SS_key=None):
    print("SS_key :" + SS_key)
    """
    Generates an academic literature review for a given research question.
    Args:
      research_question (str): The research question to generate a literature review for.
      SS_key (str, optional): The Semantic Scholar API key.
    Returns:
      str: The generated literature review.
    Examples:
      >>> literature_review('What is the impact of AI on healthcare?')
      Research question: What is the impact of AI on healthcare?
      Auto Researcher initiated!
      Generating keyword combinations...
      Keyword combinations generated!
      Fetching top 20 papers...
      Top 20 papers fetched!
      Extracting research findings from papers...
      Research findings extracted!
      Synthesizing answers...
      Literature review generated!
      Academic Literature Review: ...
      References:
      1. ...
      Keyword combinations used to search for papers: 1. AI healthcare, 2. impact AI healthcare
    """
    SemanticScholar = SemanticScholarLoader(SS_key)

    print(
        colored(
            f"Research question: {research_question}", "yellow", attrs=["bold", "blink"]
        )
    )
    print(colored("Auto Researcher initiated!", "yellow"))

    # Generate keyword combinations
    print(colored("Generating keyword combinations...", "yellow"))
    keyword_combinations = generate_keyword_combinations(research_question)
    print(colored("Keyword combinations generated!", "green"))
    print("Keyword combinations starts:")
    print(", ".join(keyword_combinations))
    print("Keyword combinations ends:")

    # Fetch the top 20 papers for the research question
    search_query = research_question
    print(colored("Fetching top 20 papers....", "yellow"))
    top_papers = SemanticScholar.fetch_and_sort_papers(
        search_query, keyword_combinations=keyword_combinations, year_range="2000-2023"
    )
    print(colored("Top 20 papers fetched!", "green"))
    # print(json.dumps(top_papers))
    return top_papers


if __name__ == "__main__":
    research_question = sys.argv[1]
    SS_key = sys.argv[2]
    email = sys.argv[3]
    fileName = sys.argv[4]

    papers = literature_review(research_question, SS_key)
    file_path = os.path.join(email, fileName + ".json")

    # Check if the directory exists, if not create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'w') as file:
        json.dump(papers, file)

