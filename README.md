# 🎓 SAT Math AI Teaching Assistant
An automated AI-powered ecosystem designed to streamline the workflow for SAT Math educators. This system integrates computer vision for question extraction and Large Language Models (LLMs) for semantic analysis and content generation.

## 🌟 Key Features
**1. Automated Question Cropping:**
* Utilizes **YOLO (You Only Look Once)** to detect and extract math questions from public exam screenshots or PDFs with high precision.

**2. Intelligent Hashtag Generation:**
* Powered by **Gemma-3** to perform deep semantic analysis. Automatically generates tags including `Topic`, `Sub-topic`, and `Difficulty Level`...etc.

**3. Question Database Integration:**
* Structured storage of questions and their associated metadata into a searchable database for efficient management.

**4. Similar Question Generation (Personalized Learning):**
* Based on a student's error patterns, the system instantly generates "Mirror Questions" (similar logic/different numbers) to facilitate targeted remediation.

## 🏗️ Development Roadmap
* **Phase 1 (Current):** Core Service Implementation - FastAPI backend & Interactive Gradio Dashboard.
* **Phase 2 (Planned):** Database structural advancement for long-term data persistence.
* **Phase 3 (Planned):** Frontend UI integration for end-user deployment.

## 🛠️ Installation
```
# 1. Create a virtual environment
conda create -n sat-ai-assistant python=3.12
conda activate sat-ai-assistant

# 2. Clone the repository
git clone https://github.com/your-username/Teaching-Assistant.git
cd Teaching-Assistant

# 3. Install dependencies (Optimized for CUDA 12.8)
pip install -r requirements.txt
```

## 🚀 Usage
The system provides two parallel interfaces for different use cases:

**1. Backend API Service (FastAPI)**
Ideal for developers and system integration. Features automated interactive documentation (Swagger UI).

* **Launch Command:**
`uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
* **How to Use:** Open your browser and navigate to `http://localhost:8000/docs`. You can test the `/analyze/single` and `/analyze/batch` endpoints directly from the UI.

**2. Interactive Dashboard (Gradio Demo)**
Ideal for Teaching Assistants for visual operation and data analytics.
* **Launch Command :**
`python gradio_app.py`
* **How to Use:** Open your browser and navigate to `http://localhost:7860`.
    * **Single Analysis:** Upload a screenshot for real-time tagging.
![image](https://hackmd.io/_uploads/HJRFRpNUWe.png)

    * **Batch Processing:** Enter a server directory path to generate a full statistical report with distribution charts.
![image](https://hackmd.io/_uploads/rJ7BTT4U-l.png)


## 📂 Project Structure (Current)
```
Teaching-Assistant/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── schemas.py       # Pydantic data models
│   └── services/
│       ├── ai_engine.py # Core AI service logic (Analysis & Stats)
│       └── prompts.py   # LLM Prompt definitions
├── gradio_app.py        # Gradio UI implementation
├── requirements.txt     # Dependency list (CUDA 12.8 specific)
└── .gitignore           # Git exclusion rules
```
