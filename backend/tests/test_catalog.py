from pathlib import Path

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.catalog import PROJECT_ROOT, build_catalog, filter_catalog
from backend.main import app

client = TestClient(app)


def test_catalog_indexes_existing_demo_families_without_importing_them():
    projects = build_catalog()
    ids = [project["id"] for project in projects]

    assert len(ids) == len(set(ids))
    assert any(project["area"] == "Artificial Intelligence" for project in projects)
    assert any(project["area"] == "Natural Language Processing" for project in projects)
    assert any(project["title"] == "Weather Forecasting" for project in projects)
    assert any(project["title"] == "Diagnostics" for project in projects)
    assert any(
        project["category"].startswith("Core NLP Tasks")
        for project in projects
    )
    handwritten_source = Path(
        "nlp/advanced_nlp_and_applications/dialogue_systems/"
        "rag_based_document_chatbot/handwritten_pdf_rag/app.py"
    )
    assert any(Path(project["source"]) == handwritten_source for project in projects) == (
        PROJECT_ROOT / handwritten_source
    ).is_file()


def test_filter_is_case_insensitive_and_restricts_area():
    projects = build_catalog()
    matches = filter_catalog(projects, query="forecast", area="Machine Learning")

    assert {project["title"] for project in matches} == {
        "Market Forecasting",
        "Weather Forecasting",
    }


def test_catalog_endpoint_searches_and_paginates():
    response = client.get("/api/catalog", params={"q": "forecast", "limit": 2})

    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert len(response.json()["items"]) == 2


def test_unknown_project_returns_not_found():
    response = client.get("/api/catalog/not-a-real-project")

    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found."


def test_feature_routes_deep_link_into_existing_python_demos():
    projects = build_catalog()
    by_title = {project["title"]: project for project in projects}

    assert by_title["Neural Networks"]["legacy_node"] == "Neural Networks"
    assert by_title["News Categorization"]["legacy_view"] == "project"
    assert by_title["News Categorization"]["legacy_node"] == "News Categorization"
    handwritten = by_title.get("Handwritten PDF Intelligence")
    if handwritten is not None:
        assert handwritten["legacy_view"] == "rag_handwritten"
        assert handwritten["legacy_node"] is None
    assert by_title["Diagnostics"]["legacy_module"] == "Mini Project"


@pytest.mark.parametrize("params", [{"limit": 0}, {"offset": -1}, {"sort": "random"}])
def test_catalog_rejects_invalid_pagination_and_sort(params):
    response = client.get("/api/catalog", params=params)

    assert response.status_code == 422
