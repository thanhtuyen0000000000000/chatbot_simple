from typing import List
import autogen
import dotenv
from autogen import ChatResult
import config
import json

dotenv.load_dotenv()


assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="I answer questions by retrieving relevant data from my data. "
                   "My response is structured in JSON with the 'question' and 'answer' keys.",
    llm_config=config.llm_config,
)


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    code_execution_config=False,
    llm_config=config.llm_config,
)


def initiate_agent_chats(question: str, answer : str) -> List[ChatResult]:
    chat_result = user_proxy.initiate_chats(
        [
            {
                "recipient": assistant,
                "message": f"Retrieve the answer for the question: '{question}' from '{answer}', "
                           "and return the result as JSON with 'question' and 'answer' keys.",
                "clear_history": True,
                "silent": False,
            }
        ]
    )
    # print (chat_result)
    
    chat_resul = chat_result[0] 
    summary = chat_resul.summary  
    
    
    summary = summary.replace("json", "").strip("`").strip()

   
    summary_dict = json.loads(summary)

    with open('answer.json', 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)

    print("Đã lưu nội dung vào file 'answer.json'")
    # # Save result to 'answer.json' directly
    # with open('answer.json', 'w') as json_file:
    #     json.dump(json.loads(summary), json_file, indent=2)

    return summary
