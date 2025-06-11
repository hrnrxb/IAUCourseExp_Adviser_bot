# 📚 IAU Course Advisor Bot (Tajrobiat Chatbot)

An interactive AI-powered chatbot that helps students at Islamic Azad University Of shiraz choose the best professors and courses — based on **real student reviews** collected from the [@IAUCourseExp](https://t.me/IAUCourseExp) Telegram channel. 🤖✨

---

## 🔍 What It Does

This bot lets students ask questions like:

> "How is Dr. Ahmadi for Data Structures?"  
> "Any reviews about Digital Logic class?"  

It instantly searches a **vectorized database** of real student experiences and returns relevant, cited feedback from the Telegram channel.

---

## 🚀 Live Demo

Try the beta version now on Hugging Face Space:  
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

### 🔧 Prerequisites

Make sure you have the following installed on your system:

- ✅ Python 3.8 or higher
- ✅ `pip` (Python package manager)
- ✅ (Optional) A virtual environment tool like `venv` or `conda` for isolated environments

---

### 🔽 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/IAU-CourseAdvisor-Bot.git
cd IAU-CourseAdvisor-Bot
📦 2. Install Dependencies
Option A: Direct install
bash
Copy
Edit
pip install -r requirements.txt
Option B: (Recommended) Using a virtual environment
bash
Copy
Edit
# Create virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate          # On macOS/Linux
venv\Scripts\activate             # On Windows

# Install required packages
pip install -r requirements.txt
▶️ 3. Run the Chatbot Locally
bash
Copy
Edit
python app.py
After a few seconds, a Gradio web interface will open in your browser (or show you a localhost link).

☁️ (Optional) Deploy on Hugging Face Spaces
To deploy this project publicly:

Fork this repo to your GitHub

Create a new Gradio Space at Hugging Face Spaces

Link your GitHub repo to the Space

Make sure app.py is your entrypoint, and requirements.txt is present

Deployment will auto-start on commit.

