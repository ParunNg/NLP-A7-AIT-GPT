from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#calling vector from local
db_file_name = '../vector-store/nlp_stanford'
emb_model_name = 'hkunlp/instructor-base'
model_id = '../models/fastchat-t5-3b-v1.0/'

prompt_template = """
    I'm your friendly AIT chatbot named AIT-GPT, here to assist new students who are still unfamiliar with our campus.
    You can ask any questions you may have about AIT's history, policies, facilities, services or any related topics.
    I'll be sure to answer you to the best of my ability!!
    Context: {context}
    Question: {question}
    Answer:
    """.strip()

PROMPT = PromptTemplate.from_template(
    template = prompt_template
)

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = emb_model_name,
    model_kwargs = {"device" : device}
)

vectordb = FAISS.load_local(
    folder_path = db_file_name,
    embeddings = embedding_model,
    index_name = 'nlp' #default index
)

retriever = vectordb.as_retriever()

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

bitsandbyte_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config = bitsandbyte_config, #caution Nvidia
    device_map = 'auto',
    load_in_8bit = True
)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

question_generator = LLMChain(
    llm = llm,
    prompt = CONDENSE_QUESTION_PROMPT,
    verbose = True
)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

memory = ConversationBufferWindowMemory(
    k=3, 
    memory_key = "chat_history",
    return_messages = True,
    output_key = 'answer'
)

chain = ConversationalRetrievalChain(
    retriever=retriever,
    question_generator=question_generator,
    combine_docs_chain=doc_chain,
    return_source_documents=True,
    memory=memory,
    verbose=True,
    get_chat_history=lambda h : h
)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        prompt_question = request.form['prompt_question']
        output = chain({"question": prompt_question})
        answer = output['answer'].replace('<pad> ', '').replace('\n', '')
        ref_list = []

        for doc in output['source_documents']:
            metadata = doc.metadata
            filename = metadata['source'].split('/')[-1]
            page_no = metadata['page'] + 1
            total_pages = metadata['total_pages']
            ref_list.append({"ref_text": f"{filename} - page {page_no}/{total_pages}",
                             "ref_link": f"{filename}#page={page_no}"})

        return render_template('home.html', prompt_question=prompt_question, answer=answer, ref_list=ref_list)

    else:
        return render_template('home.html', prompt_question="", answer=None, ref_list=None)

if __name__ == '__main__':
    app.run(debug=True)