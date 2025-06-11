import gradio as gr
import google.generativeai as genai
from my_logic import answer_question
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("models/gemini-2.0-pro-exp-02-05")

def chatbot_interface(user_question):
    return answer_question(user_question, gemini_model)

demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="مثلاً: برای مدار منطقی استاد شایگان چطوره؟", label="❓ سوال شما"),
    outputs=gr.Textbox(label="📘 پاسخ"),
    title= "🤖 ربات مشاور تجربیات انتخاب واحد",
    description="پاسخ بر اساس تجربیات واقعی دانشجویان از کانال @IAUCourseExp"
)

demo.launch()
