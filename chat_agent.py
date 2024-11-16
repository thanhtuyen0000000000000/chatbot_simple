# import json
# import pickle
# import faiss
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from typing_extensions import Annotated
# import agents

# # Load FAISS index and contents_db directly
# faiss_index = faiss.read_index('faiss_index.bin')
# with open('contents_db.pkl', 'rb') as f:
#     contents_db = pickle.load(f)

# # Load sentence transformer model
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# # Main input area for entering questions
# st.header("Chatbot Question Input")
# question = st.text_input("Enter your question:")

# if question:
#     button_enabled = st.button("Generate Answer")

#     if button_enabled:

#         # Define function to retrieve answers using FAISS index
#         def retrieve_answer(question):
#             query_embedding = model.encode([question]).astype('float32') 
#             D, I = faiss_index.search(query_embedding, k=5)  # Retrieve top match
#             best_match = contents_db[I[0][0]] if I[0][0] != -1 else "No matching answer found."
#             return best_match
        
#          # Modify `agents.initiate_agent_chats()` to accept the question and retrieve the answer
#         answer = retrieve_answer(question)
#         chat_results = agents.initiate_agent_chats(question,answer)

#         # Displaying the question and answer on the app
#         with open('answer.json', 'r', encoding='utf-8') as json_file:
#             json_data = json.load(json_file)

#         st.subheader("Question:")
#         st.write(question)

#         st.caption("Answer:")
#         st.write(json_data.get("answer", None))
        

# import json
# import pickle
# import faiss
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# import agents


# faiss_index = faiss.read_index('faiss_index.bin')
# with open('contents_db.pkl', 'rb') as f:
#     contents_db = pickle.load(f)


# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# if "conversation" not in st.session_state:
#     st.session_state.conversation = []


# def retrieve_answer(question):
#     query_embedding = model.encode([question]).astype('float32')
#     D, I = faiss_index.search(query_embedding, k=5)  # Retrieve top match
#     best_match = contents_db[I[0][0]] if I[0][0] != -1 else "No matching answer found."
#     return best_match


# st.title("Chatbot giống ChatGPT")


# question = st.text_input("Hỏi gì đó:", key="input_question")


# if question:
       
#         st.session_state.conversation.append({"sender": "user", "message": question})

     
#         answer = retrieve_answer(question)
#         agents.initiate_agent_chats(question, answer)

       
#         try:
#             with open('answer.json', 'r', encoding='utf-8') as json_file:
#                 json_data = json.load(json_file)
#                 response = json_data.get("answer", "Không tìm thấy câu trả lời trong file JSON.")
#         except FileNotFoundError:
#             response = "Lỗi: Không tìm thấy file answer.json."
#         except json.JSONDecodeError:
#             response = "Lỗi: Không thể giải mã file answer.json."

      
#         st.session_state.conversation.append({"sender": "bot", "message": response})


# st.write("### Lịch sử trò chuyện:")
# for chat in st.session_state.conversation: 
#     if chat["sender"] == "user":
#         st.markdown(f"**Bạn:** {chat['message']}")
#     else:
#         st.markdown(f"**Chatbot:** {chat['message']}")


import json
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import agents


faiss_index = faiss.read_index('faiss_index.bin')
with open('contents_db.pkl', 'rb') as f:
    contents_db = pickle.load(f)


model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


if "conversation" not in st.session_state:
    st.session_state.conversation = []


def retrieve_answer(question):
    query_embedding = model.encode([question]).astype('float32')
    D, I = faiss_index.search(query_embedding, k=5)  # Retrieve top match
    best_match = contents_db[I[0][0]] if I[0][0] != -1 else "No matching answer found."
    return best_match


st.set_page_config(
    page_title="Chatbot giống ChatGPT",
    initial_sidebar_state="expanded"
)
st.title("Chatbot giống ChatGPT")

# Input for the user's question
question = st.chat_input("Hỏi gì đó:")


if question:

    # with st.chat_message("user"):
    #     st.markdown(question)
    
 
    st.session_state.conversation.append({"role": "user", "content": question})


    answer = retrieve_answer(question)
    agents.initiate_agent_chats(question, answer)


    try:
        with open('answer.json', 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            response = json_data.get("answer", "Không tìm thấy câu trả lời trong file JSON.")
    except FileNotFoundError:
        response = "Lỗi: Không tìm thấy file answer.json."
    except json.JSONDecodeError:
        response = "Lỗi: Không thể giải mã file answer.json."


    # with st.chat_message("assistant"):
    #     st.markdown(response)
    
 
    st.session_state.conversation.append({"role": "assistant", "content": response})


for chat in st.session_state.conversation:
    if chat["role"] == "user":
        with st.chat_message("user"):
            st.markdown(chat["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(chat["content"])
