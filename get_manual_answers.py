import boto3
import tempfile
from PyPDF2 import PdfReader
import openai
import os
from dotenv import load_dotenv
from io import BytesIO
import sys

from openai_call import openai_call

load_dotenv()

# urls=sys.argv[1]
# research_question=sys.argv[2]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

s3_client = boto3.client('s3',
                         aws_access_key_id=os.getenv("aws_access_key_id", ""),
                         aws_secret_access_key=os.getenv("aws_secret_access_key", ""))


# Read the PDF file from S3 bucket
def read_pdf_from_s3(bucket_name, file_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        pdf_data = response['Body'].read()
        return pdf_data
    except Exception as e:
        print(f"Error reading PDF from S3: {e}")
        return None


# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf = PdfReader(pdf_file)
    text = ''
    for page_num in range(pdf.getNumPages()):
        page = pdf.getPage(page_num)
        text += page.extractText()
    return text


# Extract answer from research paper
def extract_answer_from_paper(bucket_name, file_keys, research_question, use_gpt4=False, temperature=0, max_tokens=150):
    answers = []
    for file_key in file_keys:
        pdf_file = read_pdf_from_s3(bucket_name, file_key)
        abstract = extract_abstract_from_pdf(pdf_file)
        citation = file_key.split('/')[-1]  # Extracting the document name as the citation
        answer = extract_answer_from_abstract(abstract, research_question, use_gpt4, temperature, max_tokens)
        if answer:
            answer_with_citation = f"{answer} SOURCE: {citation}"
            answers.append(answer_with_citation)
    return answers


# Extract abstract from PDF
def extract_abstract_from_pdf(pdf_data, page_number=0):
    pdf_stream = BytesIO(pdf_data)
    pdf = PdfReader(pdf_stream)
    if page_number < len(pdf.pages):
        page = pdf.pages[page_number]
        text = page.extract_text()
        # Find the abstract within the extracted text
        abstract_start = text.lower().find("abstract")
        if abstract_start != -1:
            abstract = text[abstract_start:]
            return abstract
        else:
            print("Abstract not found on the specified page.")
    else:
        print(f"Page number {page_number} is out of range.")
    return None


# Extract answer from abstract
def extract_answer_from_abstract(abstract, research_question, use_gpt4=False, temperature=0, max_tokens=150):
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

    prompt = extract_answer_prompt.format(
        research_question=research_question, abstract=abstract
    )
    answer = openai_call(
        prompt, use_gpt4=use_gpt4, temperature=temperature, max_tokens=max_tokens
    )

    default_answer = "No answer found."
    if answer != default_answer:
        return answer
    else:
        return None


def extract_answers(urls, research_question):
    bucket_name = os.getenv("bucketName", "")
    file_keys = urls.split(",")

    answers = extract_answer_from_paper(bucket_name, file_keys, research_question)

    return answers
