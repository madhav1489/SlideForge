# 🚀 SlideForge — AI Presentation Generator (RAG Powered)

> **Drop a topic. Get a presentation.**
> SlideForge is a full-stack AI-powered presentation engine that automatically generates structured slides using **Retrieval-Augmented Generation (RAG)**.

---

## 🌐 Live Demo

🔗 **Website:** [Add your deployed link here]
👉 Example: https://your-app-url.com

---

## 🎥 Demo Video

📽️ Watch how it works:
👉 [Add your demo video link here]
(You can upload on YouTube / Loom and paste link)

---

## ✨ Features

* 🔍 **RAG Pipeline**

  * Retrieves real data from **Wikipedia + arXiv**
* 🧠 **Smart Topic Detection**

  * Automatically adjusts slide structure
* 🧩 **Chunking & Embedding**

  * Uses semantic vector search (ChromaDB)
* 📊 **Structured Slide Generation**

  * Clean headings + bullet points
* 📤 **Export Options**

  * PPTX & PDF support
* 🎨 **Modern UI**

  * Interactive, dark-themed interface

---

## 🧠 How It Works

1. **Fetch**

   * Pulls documents from Wikipedia & arXiv
2. **Chunk**

   * Splits text into smaller chunks
3. **Embed**

   * Converts text into vectors using Sentence Transformers
4. **Forge**

   * Generates structured slides using RAG

---

## 🏗️ Tech Stack

* **Frontend:** HTML, CSS, JS
* **Backend:** FastAPI
* **RAG Framework:** LangChain
* **Vector DB:** ChromaDB
* **Embeddings:** all-MiniLM-L6-v2

---

## 📸 Screenshots

> Add your UI images here (optional)

```
![Homepage](./screenshots/home.png)
![Slides](./screenshots/slides.png)
```

---

## ⚙️ Installation & Setup

```bash
# Clone repo
git clone https://github.com/madhav1489/SlideForge.git
cd SlideForge

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirement.txt

# Run backend
uvicorn app:app --reload
```

---

## 📂 Project Structure

```
SlideForge/
│── app.py
│── main.py
│── index.html
│── notebook/
│── data/
│── chroma_db/
│── requirement.txt
```

---

## 🚧 Roadmap

* [x] RAG Pipeline
* [x] Smart Topic Detection
* [x] PPTX Export
* [ ] AI Image Generation
* [ ] Custom Themes
* [ ] Multi-language Support

---

## 👨‍💻 Author

**Madhav Sharma**
AI / ML Engineer — RAG Specialist

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
