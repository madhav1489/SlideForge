# 🚀 SlideForge — AI Presentation Generator (RAG Powered)

> **Drop a topic. Get a presentation.**
> SlideForge is a full-stack AI-powered presentation engine that automatically generates structured slides using **Retrieval-Augmented Generation (RAG)**.

---

## 🎥 Demo Video





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

<img width="1914" height="979" alt="Screenshot 2026-04-10 112723" src="https://github.com/user-attachments/assets/72767d3e-90e3-4c03-a66f-e2adec0970e7" />
<img width="1901" height="973" alt="Screenshot 2026-04-10 112741" src="https://github.com/user-attachments/assets/81a6a6d1-a28d-47b2-bb15-dcc3eda054d0" />
<img width="1900" height="979" alt="Screenshot 2026-04-10 112753" src="https://github.com/user-attachments/assets/df7ee938-cdd1-48a8-89e8-a8ba47901b95" />
<img width="1904" height="984" alt="Screenshot 2026-04-10 112808" src="https://github.com/user-attachments/assets/7db75a9d-a255-4789-9019-c52a552c4151" />
<img width="1912" height="979" alt="Screenshot 2026-04-10 112819" src="https://github.com/user-attachments/assets/06cc7570-3a77-4515-8579-a642c63c078b" />
<img width="1856" height="938" alt="Screenshot 2026-04-10 112834" src="https://github.com/user-attachments/assets/2b2e4bbb-4dce-49d2-afc1-76d7f6583a29" />
<img width="1904" height="980" alt="Screenshot 2026-04-10 112849" src="https://github.com/user-attachments/assets/20fdacb9-ddf9-4413-8ac9-fef9e6325590" />
<img width="1907" height="975" alt="Screenshot 2026-04-10 112904" src="https://github.com/user-attachments/assets/7ec1a987-7f4f-4c31-9bbc-3a96375c7072" />


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
