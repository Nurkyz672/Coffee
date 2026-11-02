from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import uvicorn
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

loader = TextLoader('coffee.txt', encoding='utf-8')
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=110,
)

chunks = text_splitter.split_documents(document)
embeddings = OllamaEmbeddings(
    model = 'nomic-embed-text',
)
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты онлайн-помощник Локального гида по кофейням Бишкека (Coffee Guide).'
            'При ответе ты должен сначала посмотреть Базу знаний:'
            "База знаний {base_knowledge}"
            'Отвечай коротко и вежливо, на языке вопроса.'
            'Отвечай ТОЛЬКО по базе знаний, ничего не придумывай.'
            'Сначала назови подходящие заведения из базы знаний с адресами (не больше трёх). '
            'Если подходят несколько филиалов одной сети, назови не больше двух. '
            'Если ни одно заведение не подходит по всем условиям, назови два заведения из базы знаний '
            'с самым высоким рейтингом и скажи, каких данных о них не хватает. '
            'Называй только заведения, которые есть в базе знаний, с их адресом. '
            'Если в базе нет вообще ничего по теме, напиши, что не знаешь.'
        ),
        (
            'human',
            '{content}'
        )
    ]
)

model_name = 'gemini-3.5-flash-lite'
print('Gemini model:', model_name)
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)


class AnswerSchema(BaseModel):
    answer: str


class QuestionOutputSchema(BaseModel):
    id: int
    answer: str
    datatime: datetime


structured_model = llm.with_structured_output(AnswerSchema)


chain = prompt | structured_model

test_app = FastAPI()

question_id = 0

class QuestionSchema(BaseModel):
    question:str

@test_app.post('/questions', response_model=QuestionOutputSchema)
async def questions(data:QuestionSchema):
    global question_id
    question_id += 1
    data = data.question.strip()

    if not data:
        raise HTTPException(status_code=400, detail = 'Мындай маалымат жок')

    new_document = vector_store.similarity_search(data, k=len(chunks))
    base_text = '\n'.join(d.page_content for d in new_document)

    result = chain.batch([{'content':data, 'base_knowledge':base_text}])[0]
    return QuestionOutputSchema(id=question_id,answer=result.answer,datatime=datetime.now())


if __name__ == '__main__':
    uvicorn.run(test_app, host='127.0.0.1', port=8000)