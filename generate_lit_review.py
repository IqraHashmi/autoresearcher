#!/usr/bin/env python3
import json
import os
import requests
from dotenv import load_dotenv
from termcolor import colored

from combine_answers import combine_answers
from get_manual_answers import extract_answers
from openai_call import openai_call

load_dotenv()

EMAIL = os.getenv("EMAIL", "")
assert EMAIL, "EMAIL environment variable is missing from .env"


def get_citation_by_doi(doi):
    """
    Retrieves a citation for a given DOI.
    Args:
      doi (str): The DOI of the citation to retrieve.
    Returns:
      str: The citation for the given DOI.
    Raises:
      ValueError: If the response is not valid JSON.
    Notes:
      Requires an email address to be set in the EMAIL environment variable.
    Examples:
      >>> get_citation_by_doi("10.1038/s41586-020-2003-7")
      "Liu, Y., Chen, X., Han, M., Li, Y., Li, L., Zhang, J., ... & Zhang, Y. (2020). A SARS-CoV-2 protein interaction map reveals targets for drug repurposing. Nature, 581(7809), 561-570."
    """
    url = f"https://api.citeas.org/product/{doi}?email={EMAIL}"
    response = requests.get(url)
    try:
        data = response.json()
        return data["citations"][0]["citation"]
    except ValueError:
        return response.text


def literature_review(research_question, top_papers=[], manualAnswers=[]):
    top_papers = top_papers + manualAnswers
    # Extract answers and from the top 20 papers
    print(colored("Extracting research findings from papers...", "yellow"))
    answers = extract_answers_from_papers(top_papers, research_question)
    print(colored("Research findings extracted!", "green"))

    # Combine answers into a concise academic literature review
    print(colored("Synthesizing answers...", "yellow"))
    literature_review = combine_answers(answers, research_question)
    print(colored("Literature review generated!", "green"))

    # Extract citations from answers and append a references list to the literature review
    citations = extract_citations(answers)
    references_list = "\n".join(
        [f"{idx + 1}. {citation}" for idx, citation in enumerate(citations)]
    )
    literature_review += "\n\nReferences:\n" + references_list

    literature_review += "resultEnd"
    # Print the academic literature review
    print(colored("Academic Literature Review:", "cyan"), literature_review, "\n")

    return literature_review


def extract_answers_from_papers(
        papers, research_question, use_gpt4=False, temperature=0, max_tokens=150
):
    """
    Extracts answers from paper abstracts.
    Args:
      papers (list): A list of papers.
      research_question (str): The research question to answer.
      use_gpt4 (bool, optional): Whether to use GPT-4 for answer extraction. Defaults to False.
      temperature (float, optional): The temperature for GPT-4 answer extraction. Defaults to 0.
      max_tokens (int, optional): The maximum number of tokens for GPT-4 answer extraction. Defaults to 150.
    Returns:
      list: A list of answers extracted from the paper abstracts.
    Examples:
      >>> extract_answers_from_papers(papers, research_question)
      ['Answer 1 SOURCE: Citation 1', 'Answer 2 SOURCE: Citation 2']
    """
    answers = []
    default_answer = "No answer found."

    extract_answer_prompt = """
    `reset`
    `no quotes`
    `no explanations`
    `no prompt`
    `no self-reference`
    `no apologies`
    `no filler`
    `just answer`

    I will give you the abstract of an academic paper. Extract the answer to this research question: {research_question} from the abstract.

    If the answer is not in the abstract, then you are only allowed to respond with 'No answer found'.

    This is the abstract: {abstract}
    """

    for paper in papers:
        abstract = paper.get("abstract", "")
        title = colored(paper.get("title", ""), "magenta", attrs=["bold"])
        if "externalIds" in paper and "DOI" in paper["externalIds"]:
            citation = get_citation_by_doi(paper["externalIds"]["DOI"])
        else:
            citation = paper["url"]
        prompt = extract_answer_prompt.format(
            research_question=research_question, abstract=abstract
        )
        answer = openai_call(
            prompt, use_gpt4=use_gpt4, temperature=temperature, max_tokens=max_tokens
        )

        print(f"Processing paper: {title}")

        answer_with_citation = f"{answer}\n{citation}"
        if answer != default_answer:
            answer_with_citation = f"{answer} SOURCE: {citation}"
            answers.append(answer_with_citation)
            print(colored(f"Answer found!", "green"))
            print(colored(f"{answer_with_citation}", "cyan"))

    return answers


def extract_citations(answers):
    """
    Extracts bibliographical citations from a list of answers.
    Args:
      answers (list): A list of strings containing answers.
    Returns:
      list: A list of strings containing bibliographical citations.
    Examples:
      >>> answers = ["This is an answer. SOURCE: Smith, J. (2020).",
      ...            "This is another answer. SOURCE: Jones, A. (2021)."]
      >>> extract_citations(answers)
      ["Smith, J. (2020)", "Jones, A. (2021)"]
    """
    citations = []
    for answer in answers:
        citation_start = answer.rfind("SOURCE: ")
        if citation_start != -1:
            citation = answer[citation_start + len("SOURCE: "):]
            citations.append(citation)
    return citations


if __name__ == "__main__":
    import sys

    research_question = sys.argv[1]
    email = sys.argv[2]
    fileName = sys.argv[3]
    urls = sys.argv[4]

    print(colored("Extracting manual Answers!", "green"))
    manualAnswers = extract_answers(urls, research_question)
    #manualAnswers = []
    file_path = os.path.join("/home/ubuntu/researchPapers", email, fileName + ".json")
    print(colored("Manual Answer extracted!", "green"))
    print(manualAnswers)
    #file_path = os.path.join(email, fileName + ".json")
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        with open(file_path, 'r') as file:
            topPapers = json.load(file)
    else:
        topPapers = []
    literature_review(research_question, topPapers, manualAnswers)
