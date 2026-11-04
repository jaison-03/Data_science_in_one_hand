import {
  Activity,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Brain,
  Check,
  ChevronDown,
  Command,
  Download,
  ExternalLink,
  FileCode2,
  Filter,
  Layers3,
  Menu,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import {
  useEffect,
  useCallback,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";
import {
  fetchCatalog,
  fetchProject,
  type CatalogResponse,
  type Project,
} from "./api";
import ArchitectureVisualization from "./components/ArchitectureVisualization";
import type { ArchitectureLayer } from "./components/architecture";

const PAGE_SIZE = 12;
const LEGACY_URL = (import.meta.env.VITE_LEGACY_URL ?? "http://localhost:8501").replace(/\/$/, "");

function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem("ai-theme") !== "light");

  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("ai-theme", dark ? "dark" : "light");
  }, [dark]);

  return { dark, toggle: () => setDark((value) => !value) };
}

function useCatalogData() {
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await fetchCatalog({ limit: 100 }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load projects.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void fetchCatalog({ limit: 100 }, controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Could not load projects.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, []);

  return { data, error, loading, retry: () => void load() };
}

function ErrorNotice({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="notice notice-error" role="alert">
      <div>
        <strong>We couldn't load the project catalog.</strong>
        <p>{message}</p>
      </div>
      <button className="button button-secondary" onClick={onRetry}>
        Retry
      </button>
    </div>
  );
}

function SkeletonCards() {
  return (
    <div className="project-grid" aria-label="Loading projects" aria-busy="true">
      {Array.from({ length: 6 }, (_, index) => (
        <div className="project-card skeleton-card" key={index}>
          <div className="skeleton skeleton-line short" />
          <div className="skeleton skeleton-line" />
          <div className="skeleton skeleton-line medium" />
        </div>
      ))}
    </div>
  );
}

function App() {
  const { data, error, loading, retry } = useCatalogData();
  const theme = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-shell">
      {mobileOpen && <button className="scrim" onClick={() => setMobileOpen(false)} aria-label="Close menu" />}
      <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
        <Link to="/" className="brand" aria-label="All In All explorer home" onClick={() => setMobileOpen(false)}>
          <span className="brand-icon"><Brain size={20} /></span>
          <span>all in all<span className="brand-dot">.</span></span>
        </Link>
        <p className="sidebar-label">WORKSPACE</p>
        <nav className="sidebar-nav" aria-label="Main navigation">
          <NavLink to="/" end onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Layers3 size={18} /> Overview
          </NavLink>
          <NavLink to="/explore" onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Search size={18} /> Explore projects
          </NavLink>
          <NavLink to="/map" onClick={() => setMobileOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Brain size={18} /> AI architecture
          </NavLink>
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className="status-dot" />
            <span>{error ? "API unavailable" : loading ? "Connecting to API" : "Catalog connected"}</span>
          </div>
          <a className="legacy-link" href={`${LEGACY_URL}/`} target="_blank" rel="noreferrer">
            <ExternalLink size={16} /> Existing Streamlit app
          </a>
          <p className="sidebar-footnote">The original Python demos remain available.</p>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <SearchBox />
          <button
            className="theme-button"
            onClick={theme.toggle}
            aria-label={`Switch to ${theme.dark ? "light" : "dark"} theme`}
            title={`Switch to ${theme.dark ? "light" : "dark"} theme`}
          >
            {theme.dark ? <Sun size={18} /> : <Moon size={18} />}
            <span>{theme.dark ? "Light" : "Dark"}</span>
          </button>
        </header>
        <main id="main-content">
          {error && !data ? <ErrorNotice message={error} onRetry={retry} /> : null}
          <Routes>
            <Route path="/" element={<Overview data={data} loading={loading} error={error} retry={retry} />} />
            <Route path="/explore" element={<Explore data={data} loading={loading} error={error} retry={retry} />} />
            <Route path="/map" element={<ArchitectureMap />} />
            <Route path="/projects/:projectId" element={<ProjectDetail data={data} loading={loading} error={error} retry={retry} />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>
        <footer className="footer">
          <span>Built around your existing Python projects</span>
          <a href="https://fastapi.tiangolo.com/" target="_blank" rel="noreferrer">FastAPI <ExternalLink size={12} /></a>
        </footer>
      </div>
    </div>
  );
}

function SearchBox() {
  const [value, setValue] = useState("");
  const navigate = useNavigate();
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const searchShortcut = event.key.toLowerCase() === "k" && (event.ctrlKey || event.metaKey);
      const slashShortcut =
        event.key === "/" &&
        !event.ctrlKey &&
        !event.metaKey &&
        !(event.target instanceof HTMLInputElement) &&
        !(event.target instanceof HTMLTextAreaElement);
      if (searchShortcut || slashShortcut) {
        event.preventDefault();
        input.current?.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    navigate(`/explore${value.trim() ? `?q=${encodeURIComponent(value.trim())}` : ""}`);
  }

  return (
    <form className="global-search" role="search" onSubmit={submit}>
      <Search size={17} aria-hidden="true" />
      <input ref={input} value={value} onChange={(event) => setValue(event.target.value)} placeholder="Search 70+ projects..." aria-label="Search all projects" />
      <button type="submit" aria-label="Submit search"><span className="shortcut"><Command size={11} /> K</span></button>
    </form>
  );
}

function PageIntro({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="page-intro">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      <p className="intro-description">{description}</p>
    </div>
  );
}

function ProjectCard({ project }: { project: Project }) {
  return (
    <Link className="project-card" to={`/projects/${encodeURIComponent(project.id)}`}>
      <div className="card-topline">
        <span className={`area-dot ${areaClass(project.area)}`} />
        <span className="card-area">{project.area}</span>
        <ArrowRight className="card-arrow" size={17} aria-hidden="true" />
      </div>
      <h3>{project.title}</h3>
      <p className="card-category">{project.category}</p>
      <span className="card-source"><FileCode2 size={14} /> Python demo</span>
    </Link>
  );
}

function areaClass(area: string) {
  if (area === "Artificial Intelligence") return "area-ai";
  if (area === "Natural Language Processing") return "area-nlp";
  return "area-ml";
}

function Overview({
  data,
  loading,
  error,
  retry,
}: {
  data: CatalogResponse | null;
  loading: boolean;
  error: string;
  retry: () => void;
}) {
  const featured = useMemo(
    () => (data?.items ?? []).filter((item) => item.area === "Natural Language Processing").slice(0, 3),
    [data],
  );
  const totals = useMemo(
    () => (data?.items ?? []).reduce<Record<string, number>>((result, item) => {
      result[item.area] = (result[item.area] ?? 0) + 1;
      return result;
    }, {}),
    [data],
  );

  return (
    <div className="page-content">
      <div className="hero">
        <div className="hero-copy">
          <p className="eyebrow">YOUR AI WORKSPACE</p>
          <h1>One place for the<br />whole <span>AI landscape.</span></h1>
          <p>Explore the real Python projects behind artificial intelligence, machine learning, and language technology.</p>
          <div className="hero-actions">
            <Link className="button button-primary" to="/explore">Explore all projects <ArrowRight size={16} /></Link>
            <Link className="button button-ghost" to="/map">See the 3D map</Link>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orbit orbit-outer" /><div className="orbit orbit-inner" />
          <div className="hero-orb"><Brain size={42} strokeWidth={1.5} /></div>
          <span className="orbit-label label-ai">AI</span><span className="orbit-label label-ml">ML</span><span className="orbit-label label-nlp">NLP</span>
        </div>
      </div>

      <section className="section-block">
        <div className="section-heading">
          <div><p className="eyebrow">THE COLLECTION</p><h2>Explore by discipline</h2></div>
          <Link className="text-link" to="/explore">View all <ArrowRight size={15} /></Link>
        </div>
        <div className="stats-grid">
          <DisciplineCard area="Artificial Intelligence" count={totals["Artificial Intelligence"]} label="AI" />
          <DisciplineCard area="Machine Learning" count={totals["Machine Learning"]} label="ML" />
          <DisciplineCard area="Natural Language Processing" count={totals["Natural Language Processing"]} label="NLP" />
        </div>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><p className="eyebrow">FROM YOUR PYTHON PROJECTS</p><h2>Language in action</h2></div>
          <Link className="text-link" to="/explore?area=Natural%20Language%20Processing">Browse NLP <ArrowRight size={15} /></Link>
        </div>
        {loading && !data ? <SkeletonCards /> : error && !data ? <ErrorNotice message={error} onRetry={retry} /> : featured.length ? (
          <div className="project-grid">{featured.map((project) => <ProjectCard key={project.id} project={project} />)}</div>
        ) : <EmptyState title="No projects to show yet" detail="Start the API and retry to load the existing project catalog." />}
      </section>

      <section className="integration-note">
        <span className="integration-icon"><Activity size={19} /></span>
        <div><strong>Your existing features are still here</strong><p>Open the original Streamlit workspace to run its Python demos, integrations, and saved data.</p></div>
        <a className="button button-secondary" href={`${LEGACY_URL}/`} target="_blank" rel="noreferrer">Open workspace <ExternalLink size={14} /></a>
      </section>
    </div>
  );
}

function DisciplineCard({ area, count, label }: { area: string; count?: number; label: string }) {
  return (
    <Link className="discipline-card" to={`/explore?area=${encodeURIComponent(area)}`}>
      <div className={`discipline-icon ${areaClass(area)}`}>{label}</div>
      <div><p>{area}</p><strong>{count === undefined ? "—" : count} <span>projects</span></strong></div>
      <ArrowRight className="discipline-arrow" size={17} />
    </Link>
  );
}

function Explore({
  data,
  loading,
  error,
  retry,
}: {
  data: CatalogResponse | null;
  loading: boolean;
  error: string;
  retry: () => void;
}) {
  const [params, setParams] = useSearchParams();
  const query = params.get("q") ?? "";
  const area = params.get("area") ?? "";
  const sort = (params.get("sort") ?? "title") as "title" | "area" | "category";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);
  const filtered = useMemo(() => (data?.items ?? []).filter((project) =>
    (!area || project.area === area) &&
    (!query || `${project.title} ${project.category} ${project.description}`.toLowerCase().includes(query.toLowerCase())),
  ).sort((first, second) =>
    first[sort].localeCompare(second[sort]) || first.title.localeCompare(second.title),
  ), [data, area, query, sort]);
  const pages = Math.ceil(filtered.length / PAGE_SIZE);
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useEffect(() => {
    if (page >= pages && page > 0) {
      const next = new URLSearchParams(params);
      next.delete("page");
      setParams(next, { replace: true });
    }
  }, [page, pages, params, setParams]);

  function updateParam(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    next.delete("page");
    setParams(next);
  }

  const exportProjects = async () => {
    if (!data) return;
    const rows = filtered;
    const header = ["Title", "Area", "Category", "Source"];
    const csv = [header, ...rows.map(({ title, area: itemArea, category, source }) => [title, itemArea, category, source])]
      .map((row) => row.map((field) => `"${field.replaceAll('"', '""')}"`).join(","))
      .join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "ai-project-catalog.csv";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="page-content">
      <PageIntro eyebrow="PROJECT DIRECTORY" title="Explore projects" description="Search and filter the real AI, machine-learning, and NLP code already in this repository." />
      <div className="filters-panel">
        <label className="filter-search"><Search size={16} /><input value={query} onChange={(event) => updateParam("q", event.target.value)} placeholder="Find a project..." aria-label="Search projects" /><span className="shortcut">/</span></label>
        <label className="filter-select"><Filter size={15} /><select value={area} onChange={(event) => updateParam("area", event.target.value)} aria-label="Filter by discipline"><option value="">All disciplines</option>{(data?.areas ?? []).map((item) => <option key={item} value={item}>{item}</option>)}</select><ChevronDown size={14} /></label>
        <label className="filter-select sort-select"><ArrowDown size={15} /><select value={sort} onChange={(event) => updateParam("sort", event.target.value)} aria-label="Sort projects"><option value="title">Name</option><option value="area">Discipline</option><option value="category">Category</option></select><ChevronDown size={14} /></label>
        <button className="button button-secondary export-button" onClick={() => void exportProjects()} disabled={!filtered.length}><Download size={15} /> Export CSV</button>
      </div>
      <div className="results-count" aria-live="polite"><span><strong>{filtered.length}</strong> {filtered.length === 1 ? "project" : "projects"}{query ? <> for <strong>“{query}”</strong></> : null}</span><span>Showing {visible.length} of {filtered.length}</span></div>
      {loading && !data ? <SkeletonCards /> : error && !data ? <ErrorNotice message={error} onRetry={retry} /> : visible.length ? (
        <div className="project-grid">{visible.map((project) => <ProjectCard key={project.id} project={project} />)}</div>
      ) : <EmptyState title="No matching projects" detail="Try a different search or clear the discipline filter." action={<button className="button button-secondary" onClick={() => setParams({})}>Clear filters</button>} />}
      {pages > 1 && <Pagination current={page} pages={pages} onPage={(nextPage) => updateParam("page", nextPage ? String(nextPage) : "")} />}
    </div>
  );
}

function Pagination({ current, pages, onPage }: { current: number; pages: number; onPage: (page: number) => void }) {
  return (
    <nav className="pagination" aria-label="Project pages">
      <button className="button button-secondary" disabled={current === 0} onClick={() => onPage(current - 1)}><ArrowLeft size={15} /> Previous</button>
      <span>Page {current + 1} of {pages}</span>
      <button className="button button-secondary" disabled={current === pages - 1} onClick={() => onPage(current + 1)}>Next <ArrowRight size={15} /></button>
    </nav>
  );
}

function ArchitectureMap() {
  const [selected, setSelected] = useState<ArchitectureLayer | null>(null);

  return (
    <div className="page-content map-page">
      <PageIntro eyebrow="INTERACTIVE MODEL" title="A universe of intelligence" description="Explore how AI, machine learning, deep learning, and language technologies build on one another. Select a glowing ring or information card to highlight each layer." />
      <ArchitectureVisualization
        onSelect={setSelected}
        selectedLayer={selected?.name ?? null}
      />
      <div className="architecture-map-footer">
        <span><i /> Nested disciplines · select a layer to explore</span>
        <Link className="text-link" to="/explore?area=Artificial%20Intelligence">Explore related projects <ArrowRight size={14} /></Link>
      </div>
    </div>
  );
}

function ProjectDetail({
  data,
  loading,
  error,
  retry,
}: {
  data: CatalogResponse | null;
  loading: boolean;
  error: string;
  retry: () => void;
}) {
  const { projectId = "" } = useParams();
  const project = data?.items.find((item) => item.id === projectId);
  const [response, setResponse] = useState<Project | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailRetry, setDetailRetry] = useState(0);

  useEffect(() => {
    if (project || loading) return;
    const controller = new AbortController();
    void fetchProject(projectId, controller.signal)
      .then(setResponse)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setDetailError(reason instanceof Error ? reason.message : "Could not load this project.");
      });
    return () => controller.abort();
  }, [detailRetry, loading, project, projectId]);

  if (loading && !data) return <div className="page-content"><SkeletonCards /></div>;
  if (error && !data) return <div className="page-content"><ErrorNotice message={error} onRetry={retry} /></div>;
  const current = project ?? response;
  if (detailError) return <div className="page-content"><ErrorNotice message={detailError} onRetry={() => { setDetailError(""); setDetailRetry((attempt) => attempt + 1); }} /></div>;
  if (!current) return <NotFound />;
  const destination = new URL("/", LEGACY_URL);
  destination.searchParams.set("module", current.legacy_module);
  if (current.legacy_view) {
    destination.searchParams.set("nlp_view", current.legacy_view);
    if (current.legacy_node) destination.searchParams.set("nlp_node", current.legacy_node);
  } else if (current.legacy_node && current.legacy_module === "AI") {
    destination.searchParams.set("ai_project", current.legacy_node);
  } else if (current.legacy_node) {
    destination.searchParams.set("node", current.legacy_node);
  }

  return (
    <div className="page-content">
      <Link className="back-link" to="/explore"><ArrowLeft size={15} /> All projects</Link>
      <div className="detail-card">
        <span className={`discipline-icon detail-icon ${areaClass(current.area)}`}><Brain size={23} /></span>
        <p className="eyebrow">{current.area} <span className="breadcrumb-divider">/</span> {current.category}</p>
        <h1>{current.title}</h1>
        <p className="detail-description">{current.description}</p>
        <div className="detail-facts">
          <span><FileCode2 size={16} /><strong>Existing Python feature</strong></span>
          <code>{current.source}</code>
        </div>
        <div className="migration-note"><Check size={17} /><p>Original Python logic and integrations are preserved. This feature still runs in the existing Streamlit workspace.</p></div>
        <a className="button button-primary detail-action" href={destination.toString()} target="_blank" rel="noreferrer">Open this feature <ExternalLink size={15} /></a>
        <p className="detail-footnote">Requires the existing app running at <code>{LEGACY_URL}</code>. Start it with <code>streamlit run app.py</code>.</p>
      </div>
    </div>
  );
}

function EmptyState({ title, detail, action }: { title: string; detail: string; action?: ReactNode }) {
  return <div className="empty-state"><span><Search size={20} /></span><h3>{title}</h3><p>{detail}</p>{action}</div>;
}

function NotFound() {
  return <div className="page-content not-found"><p className="eyebrow">NOT FOUND</p><h1>We couldn't find that page.</h1><Link className="button button-primary" to="/">Back to overview <ArrowRight size={15} /></Link></div>;
}

export default App;
