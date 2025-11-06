import streamlit as st
from card_rag import search_card
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ======================== session_state 설정 ===========================
# memory 객체는 세션에 저장
if "pre_memory" not in st.session_state:
    st.session_state["pre_memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True   # <-- 수정: return_messages (복수)
    )
    
# 화면에 출력할 대화 기록 (키를 "messages"로 통일)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요, 저는 AI Assistant 입니다."}
    ]

# ======================== model & prompt 설정 =========================== 
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """
너는 카드사 직원이야. 고객의 질의가 들어오면 context에 따라 가장 혜택이 많은 카드를 3개 추천해줘. 
context 내용에 한해서만 추천해주되, context에 없는 내용은 발설하지 말아줘. 
context를 참고한 출력 포맷은 아래와 같아.

--출력 포맷--
📌 해당란에 먼저 사용자가 어떤 카드를 원하는지 파악해서 요약본을 한 줄로 작성해줘.
💳 추천카드명
    - 추천 이유
    - 해당 카드의 혜택
💳 추천카드명
    - 추천 이유
    - 해당 카드의 혜택
💳 추천카드명
    - 추천 이유
    - 해당 카드의 혜택
"""

user_prompt = """\
아래의 사용자 question을 읽고 context를 참고하여 가장 적합한 카드(사용자가 혜택을 최대로 받을 수 있는 카드)를 추천해주세요.
    
--chat_history--
{chat_history}
    
--question--
{question}
    
--context--
{context}
"""
    
final_prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("user", user_prompt)
])
    
# 사용자 입력값을 받아 딕셔너리를 생성하는 함수 정의 (함수명 오타 수정)
def get_user_input(question):
    return {
        # memory에서 올바른 속성명 사용 (.chat_memory.messages)
        "chat_history": st.session_state["pre_memory"].chat_memory.messages,
        "question": question,
        "context": search_card()
    }

chain = RunnableLambda(get_user_input) | final_prompt | model | StrOutputParser()

# 대화내용을 명시적으로 기록해주는 함수 정의
def conversation_with_memory(question):
    # 1. 메세지 출력 공간 생성
    stream_placeholder = st.empty()
    
    # 2. 응답 생성 및 출력 (stream 사용)
    full_response = ""
    for chunk in chain.stream(question):
        full_response += chunk
        stream_placeholder.write(full_response)
        
    # 3. 사용자의 입력과 ai 응답을 memory에 명시적으로 저장
    st.session_state["pre_memory"].save_context(
        {"input": question},
        {"output": full_response}
    )
    
    # 4. session_state["messages"]에 저장할 용도로 full_response 반환
    return full_response 

# ======================== 메인화면 설정 ===========================
st.title("My GPT")

# 1. 대화 기록 출력 (messages로 통일)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# 2. 사용자 질의 작성
question = st.chat_input("사용자 입력")

# 3. 사용자 질의 저장 & 출력
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

# 4. ai 답변 생성 & 출력
if st.session_state["messages"][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        try:
            ai_response = conversation_with_memory(question)
            # st.question_state -> st.session_state, key "messages"로 append
            st.session_state["messages"].append({"role": "assistant", "content": ai_response})
        except Exception as e:
            error_ = f"""\
에러가 발생했습니다. 메세지를 다시 입력해주세요.

발생 에러: {e}
"""
            st.error(error_)
