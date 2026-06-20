# 🤖 AgenticAI — LangGraph Chatbot

A full-stack AI chatbot built with **LangGraph**, **OpenRouter (GPT-OSS 120B)**, and **FastAPI** — featuring tool-calling agents, a premium chat UI, and a real-time developer playground for visualizing graph execution.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange?logo=langchain&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-GPT--OSS_120B-blue?logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)

---

## ✨ Features

- **🧠 LangGraph Agent** — Stateful graph with conditional tool-calling loop (`LLM → tools_condition → ToolNode → LLM`)
- **🔧 Built-in Tools** — `get_current_time`, `calculate` (math expressions), `search_knowledge` (knowledge base lookup)
- **💬 Premium Chat UI** — Clean 3-column layout with sidebar navigation, live feed panel, markdown rendering, and responsive design
- **⚡ Developer Playground** — Real-time graph execution trace with node-level animations (idle → running → success/failed), expandable metadata panels, streaming token simulation, and TTFT/TPOT latency metrics
- **🔄 Tool Call Visualization** — Amber inline cards showing tool name, arguments, and collapsible JSON responses in the chat flow
- **🌊 Real-time Streaming** — Server-Sent Events (SSE) provide a progressive typewriter effect as the model generates text
- **🚀 OpenRouter-Powered** — Inference provided by OpenRouter's flexible API using GPT-OSS 120B

---

## 🏗️ Architecture

```
User Message
     │
     ▼
┌─────────────┐
│  FastAPI     │──── GET /           → Chat UI
│  Backend     │──── GET /playground → Developer Playground
│  (app.py)    │──── POST /chat      → LangGraph Agent
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────┐
│         LangGraph StateGraph         │
│                                      │
│  START → llmchatbot → tools_condition│
│              ↑            │          │
│              │      ┌─────┴─────┐    │
│              │      ▼           ▼    │
│              └── ToolNode      END   │
│          (time/calc/search)          │
└──────────────────────────────────────┘
```

---

## 📂 Project Structure

```
AgenticAI_Langraph/
├── 1-BasicChatBot/
│   ├── app.py                    # FastAPI backend + LangGraph agent
│   ├── basicchatbot.ipynb        # Jupyter notebook (learning/exploration)
│   ├── .env                      # OPENROUTER_API_KEY (not committed)
│   └── templates/
│       ├── index.html            # Premium Chat UI
│       └── playground.html       # Developer Playground (React + Tailwind)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/DHRUVRAJPUTTT/BasicChatBot_LangGraph.git
cd AgenticAI_Langraph
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure API Key

Create `1-BasicChatBot/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Get an API key at [openrouter.ai](https://openrouter.ai/)

### 3. Run

```bash
cd 1-BasicChatBot
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open

| Page | URL |
|------|-----|
| 💬 Chat UI | [http://127.0.0.1:8000](http://127.0.0.1:8000) |
| ⚡ Developer Playground | [http://127.0.0.1:8000/playground](http://127.0.0.1:8000/playground) |

---

## 🔧 Available Tools

| Tool | Description | Example Prompt |
|------|-------------|----------------|
| `get_current_time` | Returns current date & time | "What time is it?" |
| `calculate` | Evaluates math expressions (sqrt, sin, log, etc.) | "Calculate 2**10 + sqrt(144)" |
| `search_knowledge` | Queries a knowledge base | "Tell me about mitochondria" |
| `Tavily_Websearch` | Tells Realtime World Information | "Ask it any RealWolrd Live Query" |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | OpenRouter — GPT-OSS 120B |
| **Agent Framework** | LangGraph (StateGraph + ToolNode) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend (Chat)** | Vanilla HTML/CSS/JS (Inter font, responsive 3-column layout) |
| **Frontend (Playground)** | React 18 + Tailwind CSS (via CDN) |
| **Orchestration** | LangChain Core (tools, messages) |


### Chat UI
> Clean 3-column layout with sidebar, chat area, and live feed panel.

### Developer Playground
> Real-time graph execution trace with node animations, tool call cards, streaming responses, and latency metrics.


**Built by Dhruv** 🚀
