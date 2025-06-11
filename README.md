# 📚 IAU Course Advisor Bot (Tajrobiat Chatbot)

An interactive AI-powered chatbot that helps students at Islamic Azad University choose the best professors and courses — based on **real student reviews** collected from the [@IAUCourseExp](https://t.me/IAUCourseExp) Telegram channel. 🤖✨

---

## 🔍 What It Does

This bot lets students ask questions like:

> "How is Dr. Ahmadi for Data Structures?"  
> "Any reviews about Digital Logic class?"  

It instantly searches a **vectorized database** of real student experiences and returns relevant, cited feedback from the Telegram channel.

---

## 🚀 Live Demo

Try the beta version now on Hugging Face Spaces:  
👉 [Test the Bot](https://huggingface.co/spaces/IAUCourseExp/Tajrobiat_Bot)

---

## 📂 Dataset Preview

You can explore the dataset used for training the chatbot here:  
📎 [Tajrobiat Dataset (till Ordibehesht 14, 1404)](https://huggingface.co/datasets/IAUCourseExp/TajrobiatExpriences-till14ordibehest1404)

---

## 🧠 Tech Stack

- 🔎 **FAISS**: Vector similarity search engine
- 💬 **Transformers**: for embedding questions and texts
- 🧰 **Gradio UI**: to provide a user-friendly chat interface
- 🗃️ **Telegram Data**: Reviews from over 1000+ real students

---

## 🛠️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/IAU-CourseAdvisor-Bot.git
cd IAU-CourseAdvisor-Bot
