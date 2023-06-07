from openai_call import openai_call
import tiktoken

literature_review_prompt = """"
`reset`
`no quotes`
`no explanations`
`no prompt`
`no self-reference`
`no apologies`
`no filler`
`just answer`

I will give you a list of research findings and a research question.

Synthesize the list of research findings to generate a scientific literature review. Also, identify knowledge gaps and future research directions.

Make sure to always reference every research finding you use with in-text citations in APA format using the source provided. 

Only use the research findings I provide you with to create your literature review. Only give me the output and nothing else.

Now, using the concepts above, create a literature review for this research question '{research_question}' using the following research findings:

{answer_list}
"""


def combine_answers(answers, research_question, use_gpt4=False, temperature=0.1):
    """
    Combines a list of answers into a concise literature review using OpenAI API.
    Args:
      answers (list): A list of answers to combine.
      research_question (str): The research question to use in the literature review.
      use_gpt4 (bool, optional): Whether to use GPT-4 for the literature review. Defaults to False.
      temperature (float, optional): The temperature to use for the OpenAI API. Defaults to 0.1.
    Returns:
      str: The literature review.
    Examples:
      >>> answers = ["Answer 1", "Answer 2"]
      >>> research_question = "What is the impact of AI on society?"
      >>> combine_answers(answers, research_question)
      "The impact of AI on society is significant. Answer 1...Answer 2..."
    """
    answer_list = "\n\n".join(answers)

    prompt = literature_review_prompt.format(
        research_question=research_question, answer_list=answer_list
    )

    print("prompt-------------")
    print(prompt)
    print("prompt----------")
    # Calculate the tokens in the input
    input_tokens = count_tokens(prompt)
    # Calculate the remaining tokens for the response
    remaining_tokens = 4080 - input_tokens
    max_tokens = max(remaining_tokens, 0)
    literature_review = openai_call(
        prompt, use_gpt4=use_gpt4, temperature=temperature, max_tokens=max_tokens
    )

    return literature_review


def count_tokens(text):
    """
    Counts the number of tokens in a given text.
    Args:
      text (str): The text to tokenize.
    Returns:
      int: The number of tokens in `text`.
    Examples:
      >>> count_tokens("This is a sentence.")
      6
    Notes:
      The encoding used is determined by the `tiktoken.encoding_for_model` function.
    """
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    return len(tokens)
