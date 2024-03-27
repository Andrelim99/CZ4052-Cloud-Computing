from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

# from langchain.chat_models import vertexai, openai
# from langchain.chains import QAGenerationChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
# from langchain.chains import RetrievalQA
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI

import os 
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv
import ast

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

os.environ["OPENAI_API_KEY"] = ""
ques_parameters = ques_parameters = {
    # "candidate_count": 1,
    "model_name":"text-bison",
    "max_output_tokens": 2048,
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 40,
    "verbose": True,
}
question_llm = VertexAI(**ques_parameters)

# Set file path
# file_path = 'SDG.pdf'

def count_pdf_apges(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print("Error:", e)
        return None

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    question_gen = ''

    for page in documents:
        question_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
    model_name = 'gpt-3.5-turbo',
    chunk_size = 10000,
    chunk_overlap = 0
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    return document_ques_gen

def llm_pipeline(file_path):
    document_ques_gen = file_processing(file_path)

    prompt_template = """
    You are an expert at creating questions based on the content in documents.
    Your goal is to prepare a student for their exam and tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions in an MCQ format with 4 possible solutions that will prepare the students for their tests. The output specifications are as follows:
    'question', 'possible_answers', 'correct_answer', 'explanation'. Separate the questions with a *.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating practice questions based on content in documents.
    Your goal is to help a stuednt prepare for their exams and test.
    We have received some practice questions to a certain extent: {existing_answer}. The question, possible answers, correct answer and explanation can be found in the 'question', 'possible_answers', 'correct_answer' and 'explanation' fields respectively.
    We have the option to refine the existing questions or add new ones. 
    (only if necessary) with some more context below. Separate the questions with a *.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = question_llm, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Outputs a list of questions
    ques = ques_gen_chain.run(document_ques_gen)
    print(ques)
    ques_list = ques.split(",\n*\n")
    formatted_ques_list = []
    for i, q in enumerate(ques_list):
        try:
            f_q = ast.literal_eval(q)
            formatted_ques_list.append(f_q)
        except:
            print(f"Error!\n {q}")
            
    print(formatted_ques_list)

    return formatted_ques_list


def get_csv (file_path):
    formatted_ques_list = llm_pipeline(file_path)
    fields = formatted_ques_list[0].keys()
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        writer.writeheader()

        for q in formatted_ques_list:
            writer.writerow(q)
    return output_file


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)

    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
    res = Response(response_data)
    return res


@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res


# Part 2
@app.get("/upload_pdf")
async def index(request: Request):
    return templates.TemplateResponse("upload_pdf.html", {"request": request})

@app.post("/upload_pdf")
async def chat(request: Request, pdf_filename: str = Form(...)):
    ques_doc = file_processing(pdf_filename)
    # Process into text
    process_txt = ''
    for d in ques_doc:
        process_txt += d.page_content
    # print(process_txt)
    # Save text
    data = {
        "text": process_txt
    }

    with open("static/output/process_txt.json", "w") as f:
        json.dump(data, f)
    
    # Testing reading of text
    # with open("static/output/process_txt.json", "r") as f:
    #     d = json.load(f)
    # print(d['text'])

    response_data = jsonable_encoder(json.dumps({"msg": "success"}))
    res = Response(response_data)
    return res

@app.get("/upload_pdf/get_text")
async def chat(request: Request):
    print("Here")
    with open("static/output/process_txt.json", "r") as f:
        d = json.load(f)
    
    # response_data = jsonable_encoder(d)
    # res = Response(response_data)
    return d

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
