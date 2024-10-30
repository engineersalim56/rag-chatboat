import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# Inject the CSS
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        model="llama-3.1-70b-versatile",
    )

web_search_tool = TavilySearchResults(k = 3)

system_template = """You are a knowledgeable assistant tasked with providing detailed information and guidance about the Construction Industry Scheme (CIS) as it was implemented post-April 2007. The data you will be working with is structured into different sections and content areas, each dealing with specific aspects of the scheme, such as contractor registration, subcontractor verification, deductions, compliance, relevant legislation etc.

When responding to user queries, you should use the following guidelines:

0- Give link in the response too.

1- Content Structure: The information is organized into numbered sections, each covering a different topic within the CIS. Begin by identifying the relevant section or content area based on the user's query.

2- Contextual Understanding: Use the provided content to deliver accurate, concise, and contextually appropriate answers. If a user asks about a specific part of the scheme (e.g., subcontractor verification or monthly returns), focus on the relevant section and summarize key points.

3- User Guidance: If the user's question relates to where to find certain information within the CIS documentation, guide them to the appropriate section (e.g., 'For details on subcontractor registration, refer to CISR40000').

4- Handling Feedback: If the user is unsatisfied with the response or provides feedback, acknowledge their input and attempt to refine your answer by pulling from different relevant sections.

5- Cautions: Do not attempt to provide legal advice or make assumptions beyond the information provided. If the answer is not within your scope, inform the user that the query is outside your remit.

6- Clarity and Precision: Keep your responses clear and precise. If a term or concept is complex, provide a brief explanation or definition to aid understanding.
"""
prompttt = PromptTemplate(
    template = """<|begin_of_text|><|start_header_id|>systenm<|end_header_id|> You are a knowledgeable assistant tasked with providing detailed information and guidance about the Construction Industry Scheme (CIS) as it was implemented post-April 2007. The data you will be working with is structured into different sections and content areas, each dealing with specific aspects of the scheme, such as contractor registration, subcontractor verification, deductions, compliance, relevant legislation etc.
    Use the following pieces of retrien context to answer the question. If you don't know the answer, just say that you don't know.
    Deliver all information that you can delevir detailed<|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

def web_search(query):
    docs = web_search_tool.invoke({"query": query})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    
    rag_chain = prompttt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": web_results, "question": query})
    
    return generation

def checker(question):
    prompt1 = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a checker assessing whether a question
        is related to "Government Information". The data you will be working with is structured into different sections and content areas, each dealing with specific aspects of the scheme, such as contractor registration, subcontractor verification, deductions, compliance, relevant legislation etc.
        Provide the binary score as a JSON with a single key 'score' with yes or no and  no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    
    answer_grader = prompt1 | llm | JsonOutputParser()
    ans = answer_grader.invoke({"question": question})
    
    return ans['score']

def relevance_check(question, generation):
    prompt1 = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt1 | llm | JsonOutputParser()
    ans = answer_grader.invoke({"question": question, "generation": generation})
    
    return ans['score']

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

def process_query(vectordb, query, chat_history):
    # Create a retriever from the FAISS vector database
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Create a ConversationalRetrievalChain with a StuffedDocumentChain
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        verbose=True,
    )

    print("3")
    # Format chat history to a list of tuples
    formatted_chat_history = [(item['question'], item['answer']) for item in chat_history]

    print("4")
    # Run the prompt and return the response
    response = chain({"question": query, "chat_history": formatted_chat_history})

    print("5")
    return response

def get_database():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = FAISS.load_local("faisss", embeddings, allow_dangerous_deserialization=True)
    return vectordb
    

def main():
    # Set the title and subtitle of the app
    st.title("🦜🔗 Construction Industry Scheme Reforms Bot!")
    st.header("Ask Questions")
    query = st.text_input("Ask a question (query/prompt)")
    st.session_state['vectordb'] = get_database()

    if st.button("Submit Query", key="query_submit"):
        # if checker(query) == "no":
        #     st.write("I have the information related to Construction Industry Scheme (CIS). Your question is not related to that so kindly ask relevant questions only.")
        # else:
        chat_history = st.session_state.get('chat_history', [])

        if "cis" in query:
            query = query.replace("cis", "Construction Industry Scheme Reforms")
        elif "cisr" in query:
            query = query.replace("cisr", "Construction Industry Scheme Reforms")
        elif "CIS" in query:
            query = query.replace("CIS", "Construction Industry Scheme Reforms")
        elif "CISR" in query:
            query = query.replace("CISR", "Construction Industry Scheme Reforms")
        elif "Cis" in query:
            query = query.replace("Cis", "Construction Industry Scheme Reforms")
        elif "Cisr" in query:
            query = query.replace("Cisr", "Construction Industry Scheme Reforms")
        
        response = process_query(st.session_state['vectordb'], query, chat_history)
        
        check = relevance_check(query, response["answer"])
        
        if check == "no":
            print("Searching")
            search_result = web_search(query)
            st.write(search_result)
            chat_history.append({"question": query, "answer": search_result})
        else:
            print("No Searching")
            st.write(response["answer"])
            chat_history.append({"question": query, "answer": response["answer"]})
            
        st.session_state['chat_history'] = chat_history

if __name__ == "__main__":
    main()
