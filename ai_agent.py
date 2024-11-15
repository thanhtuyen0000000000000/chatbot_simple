import os
import autogen
from autogen import AssistantAgent,UserProxyAgent
import streamlit as st
import sys
import io
llm_configs = {
    "model": "gpt-3.5-turbo"
}
user_proxy = UserProxyAgent(
    "user",
    human_input_mode= "NEVER",
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE")
)

assistant = AssistantAgent(
    "assitant",
    llm_config=llm_configs
   
)

# Tạo giao diện Streamlit
st.title("Chatbot Weather Information for Ho Chi Minh City")

# Hàm hỗ trợ để lấy đầu ra từ initiate_chat
def get_initiate_chat_output(user_proxy, assistant, message):
    # Tạo một đối tượng StringIO để lưu trữ đầu ra của stdout
    old_stdout = sys.stdout  # Lưu lại stdout hiện tại
    sys.stdout = buffer = io.StringIO()  # Ghi đè stdout tạm thời

    try:
        # Thực hiện hội thoại
        user_proxy.initiate_chat(assistant, message=message,max_consecutive_auto_reply=0,is_termination_msg=lambda x: x.get("content", "").strip().endswith("TERMINATE"),)
        # Trả về kết quả từ buffer
        return buffer.getvalue()
    finally:
        sys.stdout = old_stdout  # Khôi phục stdout


# Nhận tin nhắn đầu vào từ người dùng và hiển thị kết quả
message = """Dựa vào những thông tin này: TP HCM là nơi hội tụ nhiều nền văn hóa; với các sản phẩm du lịch đa dạng; là với những hoạt động vui chơi; giải trí sôi động cả ngày lẫn đêm.TP HCM nằm trong vùng nhiệt đới gió mùa cận xích đạo. Đặc điểm chung của thời tiết ở đây là nhiệt độ cao đều trong năm; có hai mùa mưa và khô rõ rệt. Mùa mưa từ tháng 5 đến tháng 11; mùa khô từ tháng 12 đến tháng 4. Nhiệt độ trung bình khoảng 27 độ C; cao nhất lên hơn 40 độ C nhưng đa phần nắng không gay gắt; độ ẩm thấp; dịu mát về chiều tối. Nắng nóng không khắc nghiệt như thời tiết miền Bắc; nên du khách có thể ghé thăm thành phố bất kể thời điểm nào trong năm. Nếu đến vào mùa mưa; nên chuẩn bị ô để tránh những cơn mưa rào bất chợt. 
Hãy trả lời câu hỏi bằng tiếng việt: Thời tiết thành phố hồ chí minh như thế nào?"""


# Gọi hàm và lấy kết quả phản hồi từ assistant
response = get_initiate_chat_output(user_proxy, assistant, message)

# Hiển thị phản hồi từ assistant
st.write("**Assistant:**")
st.write(response)

# Tiếp tục nhận input từ người dùng
user_input = st.text_input("Gửi tin nhắn mới cho assistant:", "")

if user_input:
    # Nhận phản hồi mới và hiển thị
    response = get_initiate_chat_output(user_proxy, assistant, user_input)
    st.write("**User:**", user_input)
    st.write("**Assistant:**", response)


# Khởi tạo hội thoại và xử lý phản hồi chỉ một lần
# response = user_proxy.initiate_chat(
#     assitant,
#     message="Dựa vào những thông tin này: TP HCM là nơi hội tụ nhiều nền văn hóa; với các sản phẩm du lịch đa dạng; là với những hoạt động vui chơi; giải trí sôi động cả ngày lẫn đêm. TP HCM nằm trong vùng nhiệt đới gió mùa cận xích đạo. Đặc điểm chung của thời tiết ở đây là nhiệt độ cao đều trong năm; có hai mùa mưa và khô rõ rệt. Mùa mưa từ tháng 5 đến tháng 11; mùa khô từ tháng 12 đến tháng 4. Nhiệt độ trung bình khoảng 27 độ C; cao nhất lên hơn 40 độ C nhưng đa phần nắng không gay gắt; độ ẩm thấp; dịu mát về chiều tối. Nắng nóng không khắc nghiệt như thời tiết miền Bắc; nên du khách có thể ghé thăm thành phố bất kể thời điểm nào trong năm. Nếu đến vào mùa mưa; nên chuẩn bị ô để tránh những cơn mưa rào bất chợt. Hãy trả lời câu hỏi: Thời tiết thành phố hồ chí minh như thế nào?"
# )

# # Hiển thị câu trả lời trên Streamlit
# st.title("Thông tin thời tiết tại TP.HCM")
# st.write(response)
