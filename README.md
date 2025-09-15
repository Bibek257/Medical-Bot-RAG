# 🩺 Medical Bot RAG

A Retrieval-Augmented Generation (RAG) powered Medical Bot that provides accurate, up-to-date medical information by combining LLMs with your own medical knowledge base.

---

## 🚀 Features

- **Retrieval-Augmented Generation:** Combines LLMs with your custom medical documents for precise answers.
- **Natural Language Interface:** Ask medical questions in plain English.
- **Secure & Private:** Your data stays on your infrastructure.
- **Extensible:** Easily add or update medical documents.

---

## 🛠️ Tech Stack

- Python 3.10+
- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss) or [ChromaDB](https://www.trychroma.com/)
- Streamlit (for UI)

---

## ⚡ Quick Start

1. **Clone the repo:**

   ```bash
   git clone https://github.com/gbibek257/Medical-Bot-RAG.git
   cd Medical-Bot-RAG
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your medical documents** to the `data/` folder.

4. **Run the bot:**
   ```bash
   python app.py
   ```

---

## 📂 Project Structure

```
Medical-Bot-RAG/
├── app.py
├── data/
├── rag/
│   ├── retriever.py
│   └── generator.py
├── requirements.txt
└── README.md
```

---

## 📝 Usage

- Ask questions like:
  - "What are the symptoms of diabetes?"
  - "How is hypertension treated?"
- The bot retrieves relevant info from your documents and generates a concise answer.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

## ⚠️ Disclaimer

This bot is for informational purposes only and **not a substitute for professional medical advice**.

---

## ⭐️ Star this repo if you find it useful!
