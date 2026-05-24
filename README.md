# 🤖 DS-Research-Agent

An autonomous, LLM-powered Data Science Research Assistant designed to automate dataset exploration, execute analytical Python code, and generate comprehensive research summaries. This project showcases the implementation of an agentic loop using LangChain.

Developed as part of my Data Science Master's portfolio at **TU Dortmund**.

---

## 🚀 Core Features
* **Autonomous Reasoning:** Uses a ReAct (Reasoning and Acting) framework to process user queries.
* **Code Execution Sandbox:** Dynamically writes and safely executes Python scripts to analyze data.
* **Smart Search:** Integrates web-search capabilities to enrich data analysis with up-to-date domain contexts.
* **Automated Reporting:** Generates structured Markdown research insights and visual summaries.

---

## 🛠️ Repository Structure
```text
DS-Research-Agent/
│
├── agents/
│   └── research_agent.py   # Core agent configuration and system prompts
├── main.py                 # Application entry point 
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
