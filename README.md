# All In All — AI Project Explorer

This repository contains working Python/Streamlit demonstrations across artificial intelligence, machine learning, and natural language processing. The React/TypeScript explorer is an additional presentation layer; it discovers the actual Python project entrypoints and links to the existing Streamlit features. It does not replace live algorithms with sample results, change existing databases, or expose model/provider credentials.

## Migration strategy and current scope

Modernization is incremental so the existing demos remain runnable while each feature is migrated and tested. This first stage adds a typed FastAPI catalog API and a React/TypeScript/Vite application with responsive navigation, server-backed project discovery, filtering, sorting, pagination, CSV export, persistent theme choice, loading/error/empty states, and an interactive React Three Fiber model of the project's AI → ML → DL → NLP hierarchy. The Python entrypoints, model artifacts, integrations, and database behavior remain in place.

The catalog reports each project's source file and links to its existing Streamlit route. Feature-specific Streamlit interfaces are **not yet individually rewritten as React components or REST operations**: continue to run the legacy app to use those features. The 3D hierarchy is generated geometry, not a static image or GLB asset; no existing feature requires a loaded GLB model. No database migrations are introduced.

## Feature-to-file map

| Existing feature | Python implementation / entrypoint | New explorer surface |
| --- | --- | --- |
| Interactive AI hierarchy and AI project/category selector | [ai_mind_map.py](./ai_mind_map.py); [ai/](./ai/) | 3D model: `src/components/ArchitectureScene.tsx`; AI catalog: `/api/catalog` |
| ML teaching demos and learning categories | [mini_projects.py](./mini_projects.py) | Searchable, filterable catalog and feature detail |
| Life-expectancy, market, population, and weather dashboards | [ml/](./ml/) | Machine Learning catalog; detail links retain existing Streamlit routes |
| NLP concept map, text operations, language generation, dialogue, RAG, speech, NER, and summarization | [nlp_projects.py](./nlp_projects.py); [nlp/](./nlp/) | NLP catalog; detail links retain the existing NLP workspace |
| Agentic finance chatbot with a router, financial-education specialist, and LangGraph workflow | [workflow.py](./nlp/advanced_nlp_and_applications/dialogue_systems/finance_agentic_chatbot/workflow.py); [agents/](./nlp/advanced_nlp_and_applications/dialogue_systems/finance_agentic_chatbot/agents/); [app.py](./nlp/advanced_nlp_and_applications/dialogue_systems/finance_agentic_chatbot/app.py) | Finance Agentic AI Chatbot in the NLP Dialogue Systems catalog |
| Shared NLP applications, provider services, databases, and visualization helpers | [nlp/common/](./nlp/common/); project-local `services/`, `database/`, `core/`, and `ui/` modules | Existing Python implementations remain the source of truth |
| Original Streamlit router and four teaching pages | [app.py](./app.py); [pages/](./pages/) | Compatibility app remains available on port 8501 |
| New catalog API and API contract tests | [backend/](./backend/) | FastAPI at `/api/health` and `/api/catalog` |
| New React application, accessible routes, API client, and styling | [src/](./src/), [index.html](./index.html) | Vite dev server; desktop, tablet, and mobile layouts |

The API discovers existing `ai/**/app.py`, `nlp/**/app.py` (excluding the NLP router), four ML regression dashboards, and every named `mini_projects.py` teaching demo. Shared Python files are preserved but are not incorrectly presented as standalone applications.
Each discovered feature's catalog record maps its `source` file to the existing Streamlit destination (`legacy_module`, `legacy_node`, and `legacy_view`), providing a row-level feature-to-file map without maintaining a second, hardcoded list.

## Prerequisites

- Python 3.10+ and the repository's Python dependencies.
- Node.js 20.19+ or 22.12+ and npm 10+.

If Node.js is not installed, install the current LTS release with WinGet and open a new PowerShell terminal:

```powershell
winget install --id OpenJS.NodeJS.LTS --exact
node --version
npm --version
```

## Windows PowerShell

From the repository root, create the existing Python environment and install dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the exact JavaScript dependencies recorded in `package-lock.json`:

```powershell
npm ci
```

Run the React development server and API in **two PowerShell terminals**:

```powershell
# Terminal 1 — FastAPI catalog API, http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

```powershell
# Terminal 2 — React/Vite app, http://localhost:5173
npm run dev
```

Run the original Streamlit application in a **third terminal** to enable project feature links:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Alternatively, run just the existing application with the same Streamlit command. The new React explorer requires the FastAPI and Vite servers; project feature links also require the Streamlit server.

## Verification and production build

```powershell
npm run typecheck
npm run lint
npm test
npm run build
.\.venv\Scripts\python.exe -m pytest backend\tests
```

For a single-origin production deployment, build the frontend and start FastAPI:

```powershell
npm run build
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

FastAPI serves the compiled React application (including client-side routes) from `dist/` when it exists. The Streamlit compatibility app can continue running on port 8501.

## Configuration

The Vite development proxy sends `/api` requests to `http://127.0.0.1:8000`. To use a different Streamlit address for feature links, set `VITE_LEGACY_URL` before starting or building Vite:

```powershell
$env:VITE_LEGACY_URL = "http://localhost:8501"
npm run dev
```

The API is read-only, allows only local Vite development origins for cross-origin calls, and does not return environment variables, tokens, database contents, or model files.
