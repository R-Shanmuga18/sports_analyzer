# sports_analyzer
A multi-data Agentic RAG system that answers sports queries

(Tentative)

**Project Overview**

This project implements an Agentic RAG system over a sports dataset, designed to answer complex questions by reasoning across structured match data, unstructured documents, and live web information.
Unlike a traditional RAG pipeline, this system uses an LLM-driven decision loop that dynamically selects tools, executes multi-step reasoning, and composes grounded answers with explicit citations.
The agent is built from scratch (no high-level wrappers) to ensure full transparency, controllability, and explainability of every decision it makes — a key requirement of the assignment

**Dataset Description**

 Option B: Sports Season Data as defined in the assignment.
The dataset consists of three complementary sources:

1. Unstructured Data (Documents):
Wikipedia match reports / official season summaries
Stored as PDFs or text files
Used for:
Match narratives
Player performance explanations
Tactical insights

3. Structured Data (Tabular):
CSV dataset containing per-match statistics
Example fields:
Match date
Teams
Scores / goals
Player statistics (assists, runs, etc.)
Venue
Sourced from public datasets (e.g., Kaggle / football-data)

5. Live Web Data:
Real-time sports information via web search APIs
Used for:
Recent match results
Transfers
Injuries
Current standings
This setup satisfies the assignment requirement of combining unstructured + structured + live data sources

**Agentic Approach**

This system follows a modern Agentic RAG architecture, where reasoning is separated from retrieval and execution.
Core Idea

Instead of retrieving everything upfront, the agent:
 Understands the question
 Plans which tools to use
 Executes tools step-by-step
 Evaluates intermediate results
 Decides whether additional tools are needed
 Composes a final grounded answer

**Tools**

The agent exposes exactly three tools, each with a strict contract:

1. search_docs:
 Semantic search over match reports and season summaries
 Returns top relevant text chunks with source references

3. query_data:
 Queries structured match statistics (CSV / SQLite)
 Supports:
 SQL queries
 Natural language → query translation
 Returns tables or scalar values

5. web_search:
 Retrieves recent sports information from the web
 Returns snippets with URLs and timestamps
 Each tool is designed to do one task well, enabling reliable tool selection by the LLM .

**Tech Stack**
LLM:
 GPT-4o/groq

Backend:
 Python

Document Retrieval:
 FAISS / Chroma (vector store for semantic search)

Embeddings:
 OpenAI embeddings / equivalent

Structured Data:
 Pandas (CSV-based querying) or SQLite

Web Search:
 DuckDuckGo
 
Agent Framework:
 Custom-built agent loop (no black-box frameworks)
