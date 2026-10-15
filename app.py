import streamlit as st
import streamlit.components.v1 as components
from ai_mind_map import render_ai_workspace
from ml.supervised.regression.estimating_life_expectancy.life_expectancy import (
    render_life_expectancy_dashboard,
)
from ml.supervised.regression.market_forecasting.market_forecasting import (
    render_market_forecasting_dashboard,
)
from mini_projects import render_mini_project, render_project_launcher
from nlp_projects import render_nlp_workspace
from ml.supervised.regression.population_growth_prediction.population_growth import (
    render_population_growth_dashboard,
)
from ml.supervised.regression.weather_forecasting.weather_forecasting import (
    render_weather_forecasting_dashboard,
)

st.set_page_config(page_title="AI Architecture Explorer", page_icon="🧠", layout="wide")

# State management
if 'selected_module' not in st.session_state:
    st.session_state['selected_module'] = "Main Dashboard"

# Handle module selection via URL parameters (reliable across components)
query_params = st.query_params
if "module" in query_params:
    st.session_state['selected_module'] = query_params["module"]

selected = st.session_state['selected_module']
if selected != "NLP":
    st.title("Interactive AI Architecture Dashboard")

# HTML & SVG Diagram
interactive_svg = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: transparent; }
        svg { width: 100%; height: auto; }
        @keyframes zoom-in { from { opacity: 0; transform: scale(0.88); } to { opacity: 1; transform: scale(1); } }
        svg { transform-origin: center; animation: zoom-in 0.45s ease-out both; }
        .circle-group { cursor: pointer; transition: all 0.25s ease-in-out; }
        .circle-group:hover circle { filter: brightness(0.95); stroke-width: 4px; }
        .card-box { fill: #ffffff; rx: 12px; ry: 12px; filter: drop-shadow(0px 2px 6px rgba(0,0,0,0.08)); }
        text { user-select: none; cursor: pointer; }
        text:focus { outline: none; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }
    </style>
</head>
<body>

<svg viewBox="0 0 1200 780" xmlns="http://www.w3.org/2000/svg">
    <!-- DEFINITIONS -->
    <defs>
        <marker id="dot-blue" markerWidth="8" markerHeight="8" refX="4" refY="4"><circle cx="4" cy="4" r="4" fill="#1e88e5"/></marker>
        <marker id="dot-green" markerWidth="8" markerHeight="8" refX="4" refY="4"><circle cx="4" cy="4" r="4" fill="#43a047"/></marker>
        <marker id="dot-orange" markerWidth="8" markerHeight="8" refX="4" refY="4"><circle cx="4" cy="4" r="4" fill="#fb8c00"/></marker>
        <marker id="dot-purple" markerWidth="8" markerHeight="8" refX="4" refY="4"><circle cx="4" cy="4" r="4" fill="#8e24aa"/></marker>
    </defs>

    <!-- 1. AI CIRCLE (OUTERMOST) -->
    <g class="circle-group" data-module="AI" onclick="selectModule('AI')">
        <circle cx="420" cy="390" r="350" fill="#90caf9" stroke="#1e88e5" stroke-width="3" />
        <text x="420" y="85" text-anchor="middle" font-size="42" font-weight="800" fill="#0d47a1">AI</text>
        <text x="420" y="120" text-anchor="middle" font-size="22" font-weight="700" fill="#1565c0">Artificial Intelligence</text>
    </g>

    <!-- 2. ML CIRCLE -->
    <g class="circle-group" data-module="ML" onclick="selectModule('ML')">
        <circle cx="420" cy="440" r="260" fill="#a5d6a7" stroke="#43a047" stroke-width="3" />
        <text x="420" y="225" text-anchor="middle" font-size="38" font-weight="800" fill="#1b5e20">ML</text>
        <text x="420" y="255" text-anchor="middle" font-size="20" font-weight="700" fill="#2e7d32">Machine Learning</text>
    </g>

    <!-- 3. DL CIRCLE -->
    <g class="circle-group" data-module="DL" onclick="selectModule('DL')">
        <circle cx="420" cy="510" r="170" fill="#ffe082" stroke="#fb8c00" stroke-width="3" />
        <text x="420" y="375" text-anchor="middle" font-size="34" font-weight="800" fill="#e65100">DL</text>
        <text x="420" y="402" text-anchor="middle" font-size="18" font-weight="700" fill="#ef6c00">Deep Learning</text>
    </g>

    <!-- 4. NLP CIRCLE (INNERMOST) -->
    <g class="circle-group" data-module="NLP" onclick="selectModule('NLP')">
        <circle cx="420" cy="580" r="85" fill="#d1c4e9" stroke="#8e24aa" stroke-width="3" />
        <text x="420" y="535" text-anchor="middle" font-size="26" font-weight="800" fill="#4a148c">NLP</text>
        <text x="420" y="555" text-anchor="middle" font-size="14" font-weight="700" fill="#6a1b9a">Natural Language Processing</text>
    </g>

    <!-- CONNECTORS & SIDE CARDS -->

    <!-- AI Card -->
    <path d="M 680 140 L 780 140 L 820 100" stroke="#1e88e5" stroke-width="2" fill="none" marker-start="url(#dot-blue)" />
    <g transform="translate(820, 30)" data-module="AI" onclick="selectModule('AI')">
        <rect class="card-box" width="340" height="130" stroke="#1e88e5" stroke-width="2" />
        <text x="20" y="32" font-size="17" font-weight="700" fill="#0d47a1">AI (Artificial Intelligence)</text>
        <text x="20" y="60" font-size="13" fill="#333">• Broadest field</text>
        <text x="20" y="80" font-size="13" fill="#333">• Includes ML, DL, NLP and more</text>
        <text x="20" y="100" font-size="13" fill="#333">• Goal: human-like intelligence in machines</text>
    </g>

    <!-- ML Card -->
    <path d="M 620 280 L 780 280 L 820 270" stroke="#43a047" stroke-width="2" fill="none" marker-start="url(#dot-green)" />
    <g transform="translate(820, 200)" data-module="ML" onclick="selectModule('ML')">
        <rect class="card-box" width="340" height="130" stroke="#43a047" stroke-width="2" />
        <text x="20" y="32" font-size="17" font-weight="700" fill="#1b5e20">ML (Machine Learning)</text>
        <text x="20" y="60" font-size="13" fill="#333">• Subset of AI</text>
        <text x="20" y="80" font-size="13" fill="#333">• Learns from data</text>
        <text x="20" y="100" font-size="13" fill="#333">• Uses algorithms to make predictions/decisions</text>
    </g>

    <!-- DL Card -->
    <path d="M 560 440 L 780 440 L 820 440" stroke="#fb8c00" stroke-width="2" fill="none" marker-start="url(#dot-orange)" />
    <g transform="translate(820, 370)" data-module="DL" onclick="selectModule('DL')">
        <rect class="card-box" width="340" height="130" stroke="#fb8c00" stroke-width="2" />
        <text x="20" y="32" font-size="17" font-weight="700" fill="#e65100">DL (Deep Learning)</text>
        <text x="20" y="60" font-size="13" fill="#333">• Subset of ML</text>
        <text x="20" y="80" font-size="13" fill="#333">• Uses neural networks (multiple layers)</text>
        <text x="20" y="100" font-size="13" fill="#333">• Excels at complex patterns in data</text>
    </g>

    <!-- NLP Card -->
    <path d="M 495 580 L 780 580 L 820 610" stroke="#8e24aa" stroke-width="2" fill="none" marker-start="url(#dot-purple)" />
    <g transform="translate(820, 540)" data-module="NLP" onclick="selectModule('NLP')">
        <rect class="card-box" width="340" height="150" stroke="#8e24aa" stroke-width="2" />
        <text x="20" y="32" font-size="17" font-weight="700" fill="#4a148c">NLP (Natural Language Processing)</text>
        <text x="20" y="60" font-size="13" fill="#333">• Subset of DL (and ML)</text>
        <text x="20" y="80" font-size="13" fill="#333">• Focuses on human language</text>
        <text x="20" y="100" font-size="13" fill="#333">• Used in chatbots, translation, summarization</text>
    </g>
</svg>

<script>
    function selectModule(name, node) {
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: name
        }, "*");
        const url = new URL(window.parent.location);
        url.searchParams.set('module', name);
        if (node && name === 'ML') {
            url.searchParams.set('node', node);
        } else {
            url.searchParams.delete('node');
        }
        if (name !== 'NLP') {
            url.searchParams.delete('nlp_view');
            url.searchParams.delete('nlp_node');
        }
        window.parent.history.pushState({}, '', url);
        window.parent.location.reload();
    }

    document.querySelectorAll('svg text').forEach((label) => {
        label.setAttribute('tabindex', '0');
        label.setAttribute('role', 'button');
        label.addEventListener('click', (event) => {
            event.stopPropagation();
            const owner = label.closest('[data-module]');
            const module = owner ? owner.getAttribute('data-module') : 'ML';
            selectModule(module, label.textContent.trim());
        }, true);
        label.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                label.click();
            }
        });
    });
</script>

</body>
</html>
"""

ml_drilldown_svg = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: transparent; }
        svg { display: block; width: 100%; height: auto; transform-origin: center; animation: zoom-in 0.5s ease-out both; }
        @keyframes zoom-in { from { opacity: 0; transform: scale(0.88); } to { opacity: 1; transform: scale(1); } }
        text { user-select: none; cursor: pointer; }
        text:focus { outline: none; paint-order: stroke; stroke: #ffffff; stroke-width: 3px; }
        polygon { cursor: pointer; }
    </style>
</head>
<body>
<svg id="ml-diagram" viewBox="0 0 700 493" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="diagram-title diagram-description" data-module="ML">
    <title id="diagram-title">Machine Learning methods and applications</title>
    <desc id="diagram-description">Machine Learning branches into Unsupervised Learning, Supervised Learning, and Reinforcement Learning, with their methods and use cases.</desc>
    <g fill="none" stroke-width="4" stroke-linecap="round">
        <path d="M338 225 L239 188" stroke="#ed2939"/>
        <path d="M365 225 L465 188" stroke="#ed2939"/>
        <path d="M338 274 L338 365" stroke="#ed2939"/>
        <path d="M166 191 L139 202" stroke="#00a86b"/>
        <path d="M205 148 L205 113" stroke="#00a86b"/>
        <path d="M503 191 L540 211" stroke="#00a6d6"/>
        <path d="M478 149 L478 113" stroke="#00a6d6"/>
        <path d="M306 417 L274 429 M370 417 L408 438 M325 451 L310 477 M359 451 L368 479" stroke="#f47721" stroke-width="2"/>
        <path d="M91 212 L82 181 M79 244 L54 256 M119 263 L130 284" stroke="#67b65e" stroke-width="2"/>
        <path d="M190 50 L188 25 M220 49 L232 26 M176 67 L155 56 M234 67 L263 60" stroke="#67b65e" stroke-width="2"/>
        <path d="M458 50 L450 25 M492 50 L499 25 M450 67 L423 56 M506 67 L529 56" stroke="#71cbd3" stroke-width="2"/>
        <path d="M553 212 L600 180 M601 230 L630 212 M601 249 L630 257 M558 262 L552 285 M587 263 L597 285" stroke="#71cbd3" stroke-width="2"/>
    </g>

    <g fill="#67b65e">
        <circle cx="82" cy="181" r="4"/><circle cx="54" cy="256" r="4"/><circle cx="130" cy="284" r="4"/>
        <circle cx="188" cy="25" r="4"/><circle cx="232" cy="26" r="4"/><circle cx="155" cy="56" r="4"/><circle cx="263" cy="60" r="4"/>
    </g>
    <g fill="#71cbd3">
        <circle cx="450" cy="25" r="4"/><circle cx="499" cy="25" r="4"/><circle cx="423" cy="56" r="4"/><circle cx="529" cy="56" r="4"/>
        <circle cx="600" cy="180" r="4"/><circle cx="630" cy="212" r="4"/><circle cx="630" cy="257" r="4"/><circle cx="552" cy="285" r="4"/><circle cx="597" cy="285" r="4"/>
    </g>
    <g fill="#f47721">
        <circle cx="274" cy="429" r="5"/><circle cx="408" cy="438" r="5"/><circle cx="310" cy="477" r="5"/><circle cx="368" cy="479" r="5"/>
    </g>

    <polygon points="338,174 391,196 413,249 391,302 338,324 286,302 264,249 286,196" fill="#ed2939"/>
    <text x="338" y="242" text-anchor="middle" fill="white" font-size="27">Machine</text>
    <text x="338" y="271" text-anchor="middle" fill="white" font-size="27">Learning</text>

    <polygon points="205,135 242,150 259,188 242,226 205,241 168,226 153,188 168,150" fill="#00a86b"/>
    <text x="205" y="184" text-anchor="middle" fill="white" font-size="14">Unsupervised</text>
    <text x="205" y="201" text-anchor="middle" fill="white" font-size="14">Learning</text>

    <polygon points="478,135 515,150 531,188 515,226 478,241 441,226 425,188 441,150" fill="#00a6d6"/>
    <text x="478" y="184" text-anchor="middle" fill="white" font-size="14">Supervised</text>
    <text x="478" y="201" text-anchor="middle" fill="white" font-size="14">Learning</text>

    <polygon points="338,355 375,370 391,407 375,445 338,462 301,445 285,407 301,370" fill="#f47721"/>
    <text x="338" y="403" text-anchor="middle" fill="white" font-size="13">Reinforcement</text>
    <text x="338" y="420" text-anchor="middle" fill="white" font-size="13">Learning</text>

    <polygon points="107,195 133,206 144,233 133,260 107,271 81,260 70,233 81,206" fill="#67b65e"/>
    <text x="107" y="237" text-anchor="middle" fill="white" font-size="14">Clustering</text>
    <polygon points="205,40 231,51 242,78 231,105 205,116 179,105 168,78 179,51" fill="#67b65e"/>
    <text x="205" y="76" text-anchor="middle" fill="white" font-size="9">Dimensionality</text>
    <text x="205" y="88" text-anchor="middle" fill="white" font-size="9">Reduction</text>
    <polygon points="478,40 504,51 515,78 504,105 478,116 452,105 441,78 452,51" fill="#71cbd3"/>
    <text x="478" y="82" text-anchor="middle" fill="#164c55" font-size="10">Classification</text>
    <polygon points="576,195 602,206 613,233 602,260 576,271 550,260 539,233 550,206" fill="#71cbd3"/>
    <text x="576" y="237" text-anchor="middle" fill="#164c55" font-size="12">Regression</text>

    <g font-size="9" fill="#333">
        <text x="77" y="176" text-anchor="end">Recommender</text><text x="77" y="186" text-anchor="end">Systems</text>
        <text x="49" y="251" text-anchor="end">Targeted</text><text x="49" y="261" text-anchor="end">Marketing</text>
        <text x="139" y="282" text-anchor="start">Customer</text><text x="139" y="292" text-anchor="start">Segmentation</text>
        <text x="181" y="18" text-anchor="end">Meaningful</text><text x="181" y="28" text-anchor="end">Compression</text>
        <text x="238" y="18" text-anchor="start">Structure</text><text x="238" y="28" text-anchor="start">Discovery</text>
        <text x="149" y="51" text-anchor="end">Big data</text><text x="149" y="61" text-anchor="end">Visualisation</text>
        <text x="270" y="57" text-anchor="start">Feature</text><text x="270" y="67" text-anchor="start">Elicitation</text>
        <text x="445" y="18" text-anchor="end">Image</text><text x="445" y="28" text-anchor="end">Classification</text>
        <text x="504" y="18" text-anchor="start">Customer Retention</text>
        <text x="417" y="51" text-anchor="end">Identity Fraud</text><text x="417" y="61" text-anchor="end">Detection</text>
        <text x="536" y="56" text-anchor="start">Diagnostics</text>
        <text x="605" y="176" text-anchor="start">Advertising Popularity</text><text x="605" y="186" text-anchor="start">Prediction</text>
        <text x="635" y="209" text-anchor="start">Weather</text><text x="635" y="219" text-anchor="start">Forecasting</text>
        <text x="635" y="254" text-anchor="start">Market</text><text x="635" y="264" text-anchor="start">Forecasting</text>
        <text x="548" y="288" text-anchor="end">Population Growth</text><text x="548" y="298" text-anchor="end">Prediction</text>
        <text x="602" y="292" text-anchor="start">Estimating</text><text x="602" y="302" text-anchor="start">Life expectancy</text>
        <text x="421" y="382" text-anchor="start">Game AI</text>
        <text x="421" y="443" text-anchor="start">Skill Acquisition</text>
        <text x="268" y="433" text-anchor="end">Real-time decisions</text>
        <text x="305" y="483" text-anchor="end">Robot Navigation</text>
        <text x="379" y="485" text-anchor="start">Learning Tasks</text>
    </g>
</svg>
<script>
    function selectModule(name, node) {
        const url = new URL(window.parent.location);
        url.searchParams.set('module', name);
        if (node && (name === 'ML' || name === 'Category' || name === 'Mini Project')) {
            url.searchParams.set('node', node);
        } else {
            url.searchParams.delete('node');
        }
        window.parent.history.pushState({}, '', url);
        window.parent.location.reload();
    }
    const diagram = document.getElementById('ml-diagram');
    const labels = Array.from(diagram.querySelectorAll('text'));
    const nodeByPosition = {
        '77:176': 'Recommender Systems', '77:186': 'Recommender Systems',
        '49:251': 'Targeted Marketing', '49:261': 'Targeted Marketing',
        '139:282': 'Customer Segmentation', '139:292': 'Customer Segmentation',
        '181:18': 'Meaningful Compression', '181:28': 'Meaningful Compression',
        '238:18': 'Structure Discovery', '238:28': 'Structure Discovery',
        '149:51': 'Big Data Visualisation', '149:61': 'Big Data Visualisation',
        '270:57': 'Feature Elicitation', '270:67': 'Feature Elicitation',
        '445:18': 'Image Classification', '445:28': 'Image Classification',
        '504:18': 'Customer Retention', '417:51': 'Identity Fraud Detection',
        '417:61': 'Identity Fraud Detection', '536:56': 'Diagnostics',
        '605:176': 'Advertising Popularity Prediction', '605:186': 'Advertising Popularity Prediction',
        '635:209': 'Weather Forecasting', '635:219': 'Weather Forecasting',
        '635:254': 'Market Forecasting', '635:264': 'Market Forecasting',
        '548:288': 'Population Growth Prediction', '548:298': 'Population Growth Prediction',
        '602:292': 'Life Expectancy Prediction', '602:302': 'Life Expectancy Prediction',
        '421:382': 'Game AI', '421:443': 'Skill Acquisition',
        '268:433': 'Real-time decisions', '305:483': 'Robot Navigation',
        '379:485': 'Learning Tasks',
        '338:242': 'Machine Learning', '338:271': 'Machine Learning',
        '205:184': 'Unsupervised Learning', '205:201': 'Unsupervised Learning',
        '478:184': 'Supervised Learning', '478:201': 'Supervised Learning',
        '338:403': 'Reinforcement Learning', '338:420': 'Reinforcement Learning',
        '107:237': 'Clustering', '205:76': 'Dimensionality Reduction',
        '205:88': 'Dimensionality Reduction', '478:82': 'Classification',
        '576:237': 'Regression'
    };
    const categoryNodes = new Set([
        'Machine Learning', 'Unsupervised Learning', 'Supervised Learning',
        'Reinforcement Learning', 'Clustering', 'Dimensionality Reduction',
        'Classification', 'Regression'
    ]);
    function activateNode(target) {
        const label = target.matches('text')
            ? target
            : labels.reduce((nearest, candidate) => {
                const a = target.getBoundingClientRect();
                const b = candidate.getBoundingClientRect();
                const distance = Math.hypot(
                    a.left + a.width / 2 - b.left - b.width / 2,
                    a.top + a.height / 2 - b.top - b.height / 2
                );
                return !nearest || distance < nearest.distance
                    ? { label: candidate, distance }
                    : nearest;
            }, null).label;
        const position = `${label.getAttribute('x')}:${label.getAttribute('y')}`;
        const node = nodeByPosition[position] || label.textContent.trim();
        if (['Population Growth Prediction', 'Life Expectancy Prediction',
             'Market Forecasting', 'Weather Forecasting'].includes(node)) {
            selectModule(node);
        } else if (categoryNodes.has(node)) {
            selectModule('Category', node);
        } else {
            selectModule('Mini Project', node);
        }
    }
    diagram.querySelectorAll('text, polygon').forEach((node) => {
        node.setAttribute('tabindex', '0');
        node.setAttribute('role', 'button');
        node.addEventListener('click', (event) => {
            event.stopPropagation();
            activateNode(node);
        });
        node.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                node.click();
            }
        });
    });
</script>
</body>
</html>
"""

if selected != "Market Forecasting":
    st.session_state.pop("market_forecast_submitted", None)

if selected == "Population Growth Prediction":
    if st.button("← Back to Machine Learning"):
        st.query_params.clear()
        st.session_state['selected_module'] = "ML"
        st.query_params["module"] = "ML"
        st.rerun()
elif selected == "Life Expectancy Prediction":
    if st.button("← Back to Machine Learning"):
        st.query_params.clear()
        st.session_state['selected_module'] = "ML"
        st.query_params["module"] = "ML"
        st.rerun()
elif selected == "Market Forecasting":
    if st.button("← Back to Machine Learning"):
        st.query_params.clear()
        st.session_state['selected_module'] = "ML"
        st.query_params["module"] = "ML"
        st.rerun()
elif selected == "Weather Forecasting":
    if st.button("← Back to Machine Learning"):
        st.query_params.clear()
        st.session_state['selected_module'] = "ML"
        st.query_params["module"] = "ML"
        st.rerun()
elif selected in ("Mini Project", "Category"):
    if st.button("← Back to main diagram"):
        st.query_params.clear()
        st.session_state['selected_module'] = "Main Dashboard"
        st.rerun()
elif selected == "ML":
    if st.button("← Back to main view"):
        st.query_params.clear()
        st.session_state['selected_module'] = "Main Dashboard"
        st.rerun()
    components.html(ml_drilldown_svg, height=850, scrolling=False)
elif selected == "AI":
    if st.button("← Back to main diagram", key="ai-map-back"):
        st.query_params.clear()
        st.session_state["selected_module"] = "Main Dashboard"
        st.rerun()
    render_ai_workspace()
elif selected == "NLP":
    render_nlp_workspace()
else:
    components.html(interactive_svg, height=720, scrolling=False)

st.markdown("---")

if selected == "Population Growth Prediction":
    render_population_growth_dashboard()
elif selected == "Life Expectancy Prediction":
    render_life_expectancy_dashboard()
elif selected == "Market Forecasting":
    render_market_forecasting_dashboard()
elif selected == "Weather Forecasting":
    render_weather_forecasting_dashboard()
elif selected == "Mini Project":
    project_name = st.query_params.get("node", "")
    render_mini_project(project_name)
elif selected == "Category":
    category_name = st.query_params.get("node", "")
    st.title(category_name)
    render_project_launcher(category_name)
elif selected == "NLP":
    pass
elif selected == "AI":
    pass
else:
    col_title, col_reset = st.columns([4, 1])
    with col_title:
        st.subheader(f"Current Workspace: **{selected}**")
    with col_reset:
        if st.button("Reset View 🔄"):
            st.query_params.clear()
            st.session_state['selected_module'] = "Main Dashboard"
            st.rerun()

    if selected == "Main Dashboard":
        st.info("💡 Click on any circle (ML, DL, NLP) in the diagram above to open its dedicated workspace.")

    elif selected == "ML":
        clicked_node = st.query_params.get("node")
        if clicked_node:
            st.success(f"Selected Machine Learning node: **{clicked_node}**")
        else:
            st.success(" welcome to **Machine Learning (ML)** Page!")
        st.caption(
            "Select any label or node in the diagram to explore it. Choose "
            "Population Growth Prediction or Estimating Life Expectancy to "
            "open an interactive dashboard. Choose Weather Forecasting to "
            "compare an ML model with the Open-Meteo forecast, or Market "
            "Forecasting to compare a regression model with a persistence baseline."
        )
        if st.button("Estimating Life Expectancy"):
            st.query_params.clear()
            st.session_state["selected_module"] = "Life Expectancy Prediction"
            st.query_params["module"] = "Life Expectancy Prediction"
            st.rerun()
        if st.button("Market Forecasting"):
            st.query_params.clear()
            st.session_state["selected_module"] = "Market Forecasting"
            st.query_params["module"] = "Market Forecasting"
            st.rerun()
        if st.button("Weather Forecasting"):
            st.query_params.clear()
            st.session_state["selected_module"] = "Weather Forecasting"
            st.query_params["module"] = "Weather Forecasting"
            st.rerun()

    elif selected == "DL":
        st.warning(" welcome to **Deep Learning (DL)** Page!")
        # Put your DL specific inputs/code here
