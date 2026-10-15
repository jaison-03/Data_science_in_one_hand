"""Interactive AI concept map and project launcher."""

import runpy
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


AI_PROJECTS = {
    "Neural Networks": "machine_learning/supervised_learning/neural_networks",
    "Support Vector Machines": "machine_learning/supervised_learning/support_vector_machines",
    "Clustering Algorithms": "machine_learning/unsupervised_learning/clustering_algorithms",
    "Dimensionality Reduction": "machine_learning/unsupervised_learning/dimensionality_reduction",
    "GANs for Image Generation": "emerging_domains/generative_ai/gans_for_image_generation",
    "Large Language Models (LLMs)": "emerging_domains/large_language_models/large_language_models_llms",
    "Model Interpretation": "emerging_domains/explainable_ai_xai/model_interpretation",
    "Rule-based Explanation": "emerging_domains/explainable_ai_xai/rule_based_explanation",
    "Bias Detection": "emerging_domains/ethical_ai/bias_detection",
    "Fairness Metrics": "emerging_domains/ethical_ai/fairness_metrics",
    "Object Detection": "core_applications/computer_vision/object_detection",
    "Facial Recognition": "core_applications/computer_vision/facial_recognition",
    "Autonomous Driving": "core_applications/robotics_and_control/autonomous_driving",
    "Robot Arm Manipulation": "core_applications/robotics_and_control/robot_arm_manipulation",
    "A* Pathfinding": "fundamental_techniques/search_algorithms/a_star_pathfinding",
    "Min-Max for Games": "fundamental_techniques/search_algorithms/min_max_for_games",
    "Constraint Satisfaction": "fundamental_techniques/logic_and_reasoning/constraint_satisfaction",
    "Expert Systems": "fundamental_techniques/logic_and_reasoning/expert_systems",
}

AI_CATEGORIES = {
    "AI": list(AI_PROJECTS),
    "MACHINE LEARNING": [
        "Neural Networks", "Support Vector Machines", "Clustering Algorithms",
        "Dimensionality Reduction",
    ],
    "Supervised Learning": ["Neural Networks", "Support Vector Machines"],
    "Unsupervised Learning": ["Clustering Algorithms", "Dimensionality Reduction"],
    "CORE APPLICATIONS": [
        "Object Detection", "Facial Recognition", "Autonomous Driving",
        "Robot Arm Manipulation",
    ],
    "Computer Vision": ["Object Detection", "Facial Recognition"],
    "Robotics & Control": ["Autonomous Driving", "Robot Arm Manipulation"],
    "EMERGING DOMAINS": [
        "GANs for Image Generation", "Large Language Models (LLMs)",
        "Model Interpretation", "Rule-based Explanation", "Bias Detection",
        "Fairness Metrics",
    ],
    "Generative AI": ["GANs for Image Generation"],
    "Explainable AI (XAI)": ["Model Interpretation", "Rule-based Explanation"],
    "Ethical AI": ["Bias Detection", "Fairness Metrics"],
    "FUNDAMENTAL TECHNIQUES": [
        "A* Pathfinding", "Min-Max for Games", "Constraint Satisfaction",
        "Expert Systems",
    ],
    "Search Algorithms": ["A* Pathfinding", "Min-Max for Games"],
    "Logic & Reasoning": ["Constraint Satisfaction", "Expert Systems"],
}

AI_CONCEPT_MAP = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <style>
        html, body {
            margin: 0;
            min-height: 100%;
            overflow: hidden;
            background: #080d1b;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        svg {
            display: block;
            width: 100%;
            height: auto;
            background: #080d1b;
        }
        .stars { fill: #d8e5ff; opacity: .64; }
        .branch {
            fill: none;
            stroke-width: 7;
            stroke-linecap: round;
            stroke-linejoin: round;
            opacity: .83;
        }
        .twig {
            fill: none;
            stroke-width: 2.8;
            stroke-linecap: round;
            stroke-linejoin: round;
            opacity: .8;
        }
        .node {
            stroke-width: 2;
            filter: url(#soft-glow);
        }
        .topic {
            stroke-width: 1.7;
            filter: url(#soft-glow);
        }
        .leaf-dot {
            stroke-width: 2;
        }
        .branch-title {
            fill: #f1f4ff;
            font-size: 18px;
            font-weight: 750;
            letter-spacing: 1.35px;
        }
        .branch-subtitle {
            fill: #9ca9c9;
            font-size: 12px;
            letter-spacing: .45px;
        }
        .topic-label {
            fill: #f3f5ff;
            font-size: 14px;
            font-weight: 650;
        }
        .leaf-label {
            fill: #e7ebf8;
            font-size: 12px;
        }
        .center-title {
            fill: #fff6b5;
            font-size: 54px;
            font-weight: 850;
            letter-spacing: 3px;
        }
        .center-subtitle {
            fill: #e3e8f5;
            font-size: 16px;
            font-weight: 550;
        }
        .center-note {
            fill: #b7c3dc;
            font-size: 12px;
        }
        .connector-glow {
            fill: none;
            stroke-width: 17;
            stroke-linecap: round;
            opacity: .13;
            filter: url(#wide-glow);
        }
    </style>
</head>
<body>
<svg viewBox="0 0 1440 900" role="img" aria-labelledby="ai-map-title ai-map-description"
     xmlns="http://www.w3.org/2000/svg">
    <title id="ai-map-title">Artificial Intelligence concept map</title>
    <desc id="ai-map-description">
        An interactive, four-branch overview of fundamental techniques, machine
        learning, core applications, and emerging AI domains. Select a project
        label to open its app or a topic label to browse that category.
    </desc>
    <defs>
        <radialGradient id="map-background">
            <stop offset="0" stop-color="#111a31"/>
            <stop offset=".7" stop-color="#0b1222"/>
            <stop offset="1" stop-color="#080d1b"/>
        </radialGradient>
        <linearGradient id="center-fill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#145d70"/>
            <stop offset=".5" stop-color="#1a3659"/>
            <stop offset="1" stop-color="#432e69"/>
        </linearGradient>
        <linearGradient id="center-edge">
            <stop stop-color="#73f1e0"/>
            <stop offset=".48" stop-color="#8edfff"/>
            <stop offset="1" stop-color="#e3a8ff"/>
        </linearGradient>
        <filter id="soft-glow" x="-80%" y="-80%" width="260%" height="260%">
            <feGaussianBlur stdDeviation="5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="wide-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="12"/>
        </filter>
        <filter id="center-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="12" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
    </defs>
    <rect width="1440" height="900" fill="url(#map-background)"/>
    <g class="stars">
        <circle cx="42" cy="52" r="1.5"/><circle cx="117" cy="98" r="1"/><circle cx="235" cy="49" r="1.4"/>
        <circle cx="352" cy="92" r="1"/><circle cx="505" cy="43" r="1.4"/><circle cx="620" cy="93" r="1"/>
        <circle cx="814" cy="64" r="1.4"/><circle cx="963" cy="42" r="1"/><circle cx="1088" cy="86" r="1.5"/>
        <circle cx="1249" cy="47" r="1"/><circle cx="1379" cy="89" r="1.4"/><circle cx="68" cy="400" r="1"/>
        <circle cx="208" cy="477" r="1.4"/><circle cx="378" cy="464" r="1"/><circle cx="571" cy="380" r="1.4"/>
        <circle cx="862" cy="444" r="1"/><circle cx="1021" cy="402" r="1.5"/><circle cx="1208" cy="473" r="1"/>
        <circle cx="1384" cy="415" r="1.4"/><circle cx="48" cy="724" r="1.3"/><circle cx="221" cy="800" r="1"/>
        <circle cx="481" cy="844" r="1.3"/><circle cx="713" cy="821" r="1"/><circle cx="946" cy="852" r="1.4"/>
        <circle cx="1190" cy="801" r="1"/><circle cx="1374" cy="735" r="1.4"/>
    </g>

    <g aria-hidden="true">
        <path class="connector-glow" stroke="#39ddd1" d="M638 397 C585 367 555 312 501 286"/>
        <path class="connector-glow" stroke="#e14eb7" d="M802 397 C855 367 884 312 939 286"/>
        <path class="connector-glow" stroke="#9a78ed" d="M638 490 C583 520 554 584 501 617"/>
        <path class="connector-glow" stroke="#e8b34d" d="M802 490 C857 520 886 584 939 617"/>
        <path class="branch" stroke="#39ddd1" d="M638 397 C585 367 555 312 501 286"/>
        <path class="branch" stroke="#e14eb7" d="M802 397 C855 367 884 312 939 286"/>
        <path class="branch" stroke="#9a78ed" d="M638 490 C583 520 554 584 501 617"/>
        <path class="branch" stroke="#e8b34d" d="M802 490 C857 520 886 584 939 617"/>
    </g>

    <g aria-label="Fundamental techniques">
        <path class="twig" stroke="#39ddd1" d="M501 286 C461 259 433 225 393 215 M501 286 C460 298 435 329 393 340 M393 215 H294 M393 340 H294"/>
        <path class="twig" stroke="#39ddd1" d="M294 215 H88 M294 215 V250 H88 M294 340 H88 M294 340 V375 H88"/>
        <circle class="leaf-dot" cx="88" cy="215" r="5" fill="#39ddd1" stroke="#a4fff4"/>
        <circle class="leaf-dot" cx="88" cy="250" r="5" fill="#39ddd1" stroke="#a4fff4"/>
        <circle class="leaf-dot" cx="88" cy="340" r="5" fill="#39ddd1" stroke="#a4fff4"/>
        <circle class="leaf-dot" cx="88" cy="375" r="5" fill="#39ddd1" stroke="#a4fff4"/>
        <rect x="318" y="252" width="226" height="69" rx="21" fill="#123e4a" stroke="#39ddd1" class="node"/>
        <text x="431" y="279" text-anchor="middle" class="branch-title">FUNDAMENTAL</text>
        <text x="431" y="302" text-anchor="middle" class="branch-title">TECHNIQUES</text>
        <rect x="325" y="190" width="143" height="49" rx="16" fill="#16414a" stroke="#39ddd1" class="topic"/>
        <text x="396" y="220" text-anchor="middle" class="topic-label">Search Algorithms</text>
        <rect x="322" y="316" width="147" height="49" rx="16" fill="#16414a" stroke="#39ddd1" class="topic"/>
        <text x="395" y="346" text-anchor="middle" class="topic-label">Logic &amp; Reasoning</text>
        <text x="277" y="219" text-anchor="end" class="leaf-label">A* Pathfinding</text>
        <text x="277" y="254" text-anchor="end" class="leaf-label">Min-Max for Games</text>
        <text x="277" y="344" text-anchor="end" class="leaf-label">Constraint Satisfaction</text>
        <text x="277" y="379" text-anchor="end" class="leaf-label">Expert Systems</text>
    </g>

    <g aria-label="Machine learning">
        <path class="twig" stroke="#e14eb7" d="M939 286 C979 259 1007 225 1047 215 M939 286 C980 298 1005 329 1047 340 M1047 215 H1148 M1047 340 H1148"/>
        <path class="twig" stroke="#e14eb7" d="M1148 215 H1351 M1148 215 V250 H1351 M1148 340 H1351 M1148 340 V375 H1351"/>
        <circle class="leaf-dot" cx="1351" cy="215" r="5" fill="#e14eb7" stroke="#ffb3e7"/>
        <circle class="leaf-dot" cx="1351" cy="250" r="5" fill="#e14eb7" stroke="#ffb3e7"/>
        <circle class="leaf-dot" cx="1351" cy="340" r="5" fill="#e14eb7" stroke="#ffb3e7"/>
        <circle class="leaf-dot" cx="1351" cy="375" r="5" fill="#e14eb7" stroke="#ffb3e7"/>
        <polygon points="939,229 996,258 996,314 939,343 882,314 882,258" fill="#562747" stroke="#e14eb7" class="node"/>
        <text x="939" y="277" text-anchor="middle" class="branch-title">MACHINE</text>
        <text x="939" y="302" text-anchor="middle" class="branch-title">LEARNING</text>
        <rect x="972" y="190" width="149" height="49" rx="16" fill="#49243f" stroke="#e14eb7" class="topic"/>
        <text x="1046" y="220" text-anchor="middle" class="topic-label">Supervised Learning</text>
        <rect x="972" y="316" width="149" height="49" rx="16" fill="#49243f" stroke="#e14eb7" class="topic"/>
        <text x="1046" y="346" text-anchor="middle" class="topic-label">Unsupervised Learning</text>
        <text x="1165" y="219" class="leaf-label">Neural Networks</text>
        <text x="1165" y="254" class="leaf-label">Support Vector Machines</text>
        <text x="1165" y="344" class="leaf-label">Clustering Algorithms</text>
        <text x="1165" y="379" class="leaf-label">Dimensionality Reduction</text>
    </g>

    <g aria-label="Core applications">
        <path class="twig" stroke="#9a78ed" d="M501 617 C461 644 433 679 393 688 M501 617 C460 606 435 574 393 563 M393 563 H294 M393 688 H294"/>
        <path class="twig" stroke="#9a78ed" d="M294 563 H88 M294 563 V598 H88 M294 688 H88 M294 688 V723 H88"/>
        <circle class="leaf-dot" cx="88" cy="563" r="5" fill="#9a78ed" stroke="#d0baff"/>
        <circle class="leaf-dot" cx="88" cy="598" r="5" fill="#9a78ed" stroke="#d0baff"/>
        <circle class="leaf-dot" cx="88" cy="688" r="5" fill="#9a78ed" stroke="#d0baff"/>
        <circle class="leaf-dot" cx="88" cy="723" r="5" fill="#9a78ed" stroke="#d0baff"/>
        <polygon points="501,560 552,586 552,638 501,664 450,638 450,586" fill="#342b60" stroke="#9a78ed" class="node"/>
        <text x="501" y="608" text-anchor="middle" class="branch-title">CORE</text>
        <text x="501" y="633" text-anchor="middle" class="branch-title">APPLICATIONS</text>
        <rect x="321" y="538" width="147" height="49" rx="16" fill="#302752" stroke="#9a78ed" class="topic"/>
        <text x="394" y="568" text-anchor="middle" class="topic-label">Computer Vision</text>
        <rect x="321" y="663" width="147" height="49" rx="16" fill="#302752" stroke="#9a78ed" class="topic"/>
        <text x="394" y="693" text-anchor="middle" class="topic-label">Robotics &amp; Control</text>
        <text x="277" y="567" text-anchor="end" class="leaf-label">Object Detection</text>
        <text x="277" y="602" text-anchor="end" class="leaf-label">Facial Recognition</text>
        <text x="277" y="692" text-anchor="end" class="leaf-label">Autonomous Driving</text>
        <text x="277" y="727" text-anchor="end" class="leaf-label">Robot Arm Manipulation</text>
    </g>

    <g aria-label="Emerging domains">
        <path class="twig" stroke="#e8b34d" d="M939 617 C979 644 1007 679 1047 688 M939 617 C980 606 1005 574 1047 563 M1047 563 H1148 M1047 688 H1148"/>
        <path class="twig" stroke="#e8b34d" d="M1148 563 H1351 M1148 563 V598 H1351 M1148 688 H1351 M1148 688 V723 H1351"/>
        <circle class="leaf-dot" cx="1351" cy="563" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <circle class="leaf-dot" cx="1351" cy="598" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <circle class="leaf-dot" cx="1351" cy="688" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <circle class="leaf-dot" cx="1351" cy="723" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <polygon points="939,560 990,586 990,638 939,664 888,638 888,586" fill="#514022" stroke="#e8b34d" class="node"/>
        <text x="939" y="608" text-anchor="middle" class="branch-title">EMERGING</text>
        <text x="939" y="633" text-anchor="middle" class="branch-title">DOMAINS</text>
        <rect x="972" y="538" width="149" height="49" rx="16" fill="#49391f" stroke="#e8b34d" class="topic"/>
        <text x="1046" y="568" text-anchor="middle" class="topic-label">Generative AI</text>
        <rect x="972" y="663" width="149" height="49" rx="16" fill="#49391f" stroke="#e8b34d" class="topic"/>
        <text x="1046" y="693" text-anchor="middle" class="topic-label">Explainable AI (XAI)</text>
        <text x="1165" y="567" class="leaf-label">GANs for Image Generation</text>
        <text x="1165" y="602" class="leaf-label">Large Language Models (LLMs)</text>
        <text x="1165" y="692" class="leaf-label">Model Interpretation</text>
        <text x="1165" y="727" class="leaf-label">Rule-based Explanation</text>
        <path class="twig" stroke="#e8b34d" d="M939 664 V777 H1047"/>
        <rect x="972" y="752" width="149" height="49" rx="16" fill="#49391f" stroke="#e8b34d" class="topic"/>
        <text x="1046" y="782" text-anchor="middle" class="topic-label">Ethical AI</text>
        <path class="twig" stroke="#e8b34d" d="M1121 777 H1148 V766 H1351 M1148 777 V805 H1351"/>
        <circle class="leaf-dot" cx="1351" cy="766" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <circle class="leaf-dot" cx="1351" cy="805" r="5" fill="#e8b34d" stroke="#ffdfa0"/>
        <text x="1165" y="770" class="leaf-label">Bias Detection</text>
        <text x="1165" y="809" class="leaf-label">Fairness Metrics</text>
    </g>

    <g filter="url(#center-glow)">
        <polygon points="720,315 805,363 805,465 720,513 635,465 635,363"
                 fill="url(#center-fill)" stroke="url(#center-edge)" stroke-width="4"/>
        <polygon points="720,329 792,370 792,458 720,499 648,458 648,370"
                 fill="none" stroke="#a8e7ff" stroke-opacity=".48" stroke-width="1.5"/>
        <text x="720" y="419" text-anchor="middle" class="center-title">AI</text>
        <text x="720" y="449" text-anchor="middle" class="center-subtitle">ARTIFICIAL INTELLIGENCE</text>
        <text x="720" y="473" text-anchor="middle" class="center-note">A connected field of ideas and applications</text>
    </g>
    <text x="720" y="865" text-anchor="middle" fill="#74809e" font-size="12" letter-spacing="2">
        AI CONCEPT MAP
    </text>
</svg>
<script>
    const projectLabels = {
        "Neural Networks": "Neural Networks",
        "Support Vector Machines": "Support Vector Machines",
        "Clustering Algorithms": "Clustering Algorithms",
        "Dimensionality Reduction": "Dimensionality Reduction",
        "GANs for Image Generation": "GANs for Image Generation",
        "Large Language Models (LLMs)": "Large Language Models (LLMs)",
        "Model Interpretation": "Model Interpretation",
        "Rule-based Explanation": "Rule-based Explanation",
        "Bias Detection": "Bias Detection",
        "Fairness Metrics": "Fairness Metrics",
        "Object Detection": "Object Detection",
        "Facial Recognition": "Facial Recognition",
        "Autonomous Driving": "Autonomous Driving",
        "Robot Arm Manipulation": "Robot Arm Manipulation",
        "A* Pathfinding": "A* Pathfinding",
        "Min-Max for Games": "Min-Max for Games",
        "Constraint Satisfaction": "Constraint Satisfaction",
        "Expert Systems": "Expert Systems"
    };
    const categoryLabels = {
        "FUNDAMENTAL": "FUNDAMENTAL TECHNIQUES",
        "TECHNIQUES": "FUNDAMENTAL TECHNIQUES",
        "Search Algorithms": "Search Algorithms",
        "Logic & Reasoning": "Logic & Reasoning",
        "MACHINE": "MACHINE LEARNING",
        "LEARNING": "MACHINE LEARNING",
        "Supervised Learning": "Supervised Learning",
        "Unsupervised Learning": "Unsupervised Learning",
        "CORE": "CORE APPLICATIONS",
        "APPLICATIONS": "CORE APPLICATIONS",
        "Computer Vision": "Computer Vision",
        "Robotics & Control": "Robotics & Control",
        "EMERGING": "EMERGING DOMAINS",
        "DOMAINS": "EMERGING DOMAINS",
        "Generative AI": "Generative AI",
        "Explainable AI (XAI)": "Explainable AI (XAI)",
        "Ethical AI": "Ethical AI",
        "ARTIFICIAL INTELLIGENCE": "AI",
        "AI": "AI"
    };
    function openNode(label) {
        const url = new URL(window.parent.location.href);
        url.searchParams.set("module", "AI");
        url.searchParams.delete("node");
        url.searchParams.delete("ai_project");
        url.searchParams.delete("ai_category");
        if (projectLabels[label]) {
            url.searchParams.set("ai_project", projectLabels[label]);
        } else if (categoryLabels[label]) {
            url.searchParams.set("ai_category", categoryLabels[label]);
        } else {
            return;
        }
        window.parent.history.pushState({}, "", url);
        window.parent.location.reload();
    }
    document.querySelectorAll("svg text").forEach((label) => {
        label.setAttribute("tabindex", "0");
        label.setAttribute("role", "button");
        label.style.cursor = "pointer";
        label.addEventListener("click", (event) => {
            event.stopPropagation();
            openNode(label.textContent.trim());
        });
        label.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                openNode(label.textContent.trim());
            }
        });
    });
</script>
</body>
</html>
"""


def render_ai_mind_map():
    """Render the interactive AI overview."""
    components.html(AI_CONCEPT_MAP, height=765, scrolling=False)


def render_ai_workspace():
    """Render a selected AI project or category inside the main dashboard."""
    project_name = st.query_params.get("ai_project")
    category_name = st.query_params.get("ai_category")

    if project_name:
        project_folder = AI_PROJECTS.get(project_name)
        if project_folder is None:
            st.error(f"Unknown AI project: {project_name}")
            return
        if st.button("← Back to AI concept map", key="ai-project-back"):
            st.query_params.pop("ai_project", None)
            st.rerun()
        project_file = Path(__file__).resolve().parent / "ai" / project_folder / "app.py"
        st.session_state["_embedded_ai_project"] = True
        try:
            runpy.run_path(str(project_file), run_name="_embedded_ai_project")
        finally:
            st.session_state.pop("_embedded_ai_project", None)
        return

    if category_name:
        project_names = AI_CATEGORIES.get(category_name)
        if project_names is None:
            st.error(f"Unknown AI category: {category_name}")
            return
        if st.button("← Back to AI concept map", key="ai-category-back"):
            st.query_params.pop("ai_category", None)
            st.rerun()
        st.subheader(category_name.title())
        st.caption("Choose a project to open it in this dashboard.")
        for project in project_names:
            if st.button(project, key=f"ai-project-{project}"):
                st.query_params.pop("ai_category", None)
                st.query_params["ai_project"] = project
                st.rerun()
        return

    render_ai_mind_map()
