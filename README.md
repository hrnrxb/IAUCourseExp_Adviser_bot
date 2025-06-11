# 📚 IAU Course Advisor Bot (Tajrobiat Chatbot)

An interactive AI-powered chatbot that helps students at Islamic Azad University Of shiraz choose the best professors and courses — based on **real student reviews** collected from the [@IAUCourseExp](https://t.me/IAUCourseExp) Telegram channel. 🤖✨

---

## 🔍 What It Does

This bot lets students ask question**S** **like**:

> "How is Dr. X for Data Structures?"
> 
> "Any reviews about Digital Logic class?"
> 
> "Sort best to worst professors of fuzzy system for me"  

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

This project integrates modern NLP and interactive web technologies to address a real-world gap: **lack of centralized and accessible student feedback** during course registration at universities.

---

- 🔎 **FAISS (Facebook AI Similarity Search)**  
Used to build a fast, scalable vector search engine for student reviews. Traditional keyword search (like Ctrl+F) is not effective for understanding **semantic similarity** in informal, Persian-language Telegram comments. FAISS enables approximate nearest neighbor search over dense sentence embeddings, providing highly relevant matches even with vague user questions.

- 💬 **Transformers (Sentence Embeddings with BERT-like models)**  
To turn both user questions and student reviews into comparable vector representations, we use pretrained language models from Hugging Face Transformers. This allows the chatbot to semantically understand queries like _"How is Dr. X for Data Structures?"_ and find matching content, even if phrased differently.

- 🧰 **Gradio UI**  
Gradio provides an intuitive and lightweight interface for interacting with the chatbot. It allows students to easily input questions and receive answers in a clean web app — without needing to install anything locally or know how the backend works. Perfect for public testing and non-technical users.

- 🗃️ **Telegram Data (Curated Student Reviews)**  
The dataset consists of real-world experiences from students at Islamic Azad University of Shiraz, collected and organized from a public Telegram channel with over 1000 messages. Each review includes metadata like course name, instructor name, and sometimes emotional tone, making it ideal for **real-life NLP prototyping** in low-resource Persian contexts.

This combination of tools allows the bot to act as a **semantic bridge** between vague human queries and fragmented student feedback — filling a critical gap in how students access peer-generated course evaluations during unit selection periods.

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
```

### 📦 2. Install Dependencies

#### Option A: Direct install

```bash
pip install -r requirements.txt
```

#### Option B: (Recommended) Using a virtual environment

##### Create virtual environment  

```bash
python -m venv venv  
```

##### Activate the environment  

```bash
source venv/bin/activate          # On macOS/Linux
```
```bash  
venv\Scripts\activate             # On Windows  
```

##### Install required packages  

```bash
pip install -r requirements.txt   
```

### ▶️ 3. Run the Chatbot Locally

```bash
python app.py   
```

After a few seconds, a Gradio web interface will open in your browser (or show you a localhost link).

### ☁️ (Optional) Deploy on Hugging Face Spaces

To deploy this project publicly:

1.  Fork this repo to your GitHub
    
2.  Create a new **Gradio Space** at [Hugging Face Spaces](https://huggingface.co/spaces)
    
3.  Link your GitHub repo to the Space
    
4.  Make sure app.py is your entrypoint, and requirements.txt is present
    

Deployment will auto-start on commit.
