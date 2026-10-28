import React, { useCallback, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Activity, AlertCircle, ArrowUpRight, Bot, Check, CheckCircle2, ChevronDown,
  ChevronRight, Circle, Clock3, Code2, FileCode2, GitBranch, History,
  LoaderCircle, Menu, MessageSquareText, Minus, Plus, RotateCcw, Search,
  ShieldCheck, Sparkles, Trash2, X,
} from 'lucide-react'
import './style.css'

const API = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001').replace(/\/$/, '')
const terminal = new Set(['completed', 'rejected', 'tests_failed', 'apply_failed', 'rolled_back', 'rollback_failed', 'investigated', 'failed'])

async function api(path, options = {}) {
  let response
  try {
    response = await fetch(`${API}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
    })
  } catch {
    throw new Error('Buildmate could not connect to the backend. Make sure the FastAPI server is running on port 8001.')
  }
  let body = {}
  try { body = await response.json() } catch { /* Empty response */ }
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : body.detail?.message || ''
    const code = body.detail?.code
    const friendly = {
      OLLAMA_UNAVAILABLE: 'Could not reach Ollama. Make sure it is running, then retry indexing.',
      OLLAMA_TIMEOUT: 'Ollama took too long generating embeddings. The indexer retries automatically; please retry indexing.',
      EMBEDDING_FAILED: 'Ollama returned invalid embeddings. Check the configured embedding model and retry.',
      CHROMA_FAILED: 'The local vector index could not be updated. Check available disk space and backend logs, then retry.',
      PROJECT_PATH_INVALID: 'That project path is unavailable. Enter an existing folder accessible to the backend.',
    }
    if (code && friendly[code]) throw new Error(friendly[code])
    if (response.status === 404) throw new Error(detail || 'The requested project or task was not found.')
    if (response.status === 503) throw new Error(detail || 'A local AI service is unavailable. Check Ollama and the configured model.')
    if (response.status >= 500) throw new Error(detail || 'Buildmate could not complete the request. Try again or inspect the task details.')
    throw new Error(detail || `The request could not be completed (${response.status}).`)
  }
  return body
}

function relativeTime(value) {
  if (!value) return 'Just now'
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000))
  if (seconds < 60) return 'Just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} hr ago`
  return `${Math.floor(seconds / 86400)} days ago`
}

function Status({ value, children }) {
  const normalized = (value || 'unknown').toLowerCase()
  const good = ['indexed', 'connected', 'completed', 'passed', 'applied'].includes(normalized)
  const waiting = ['awaiting_approval', 'pending', 'indexing', 'testing', 'investigating', 'generating_proposal', 'applying', 'running'].includes(normalized)
  const failed = ['failed', 'rejected', 'tests_failed', 'apply_failed', 'rollback_failed'].includes(normalized)
  const Icon = good ? CheckCircle2 : waiting ? Clock3 : failed ? AlertCircle : Circle
  const label = children || statusLabel(normalized)
  return <span className={`status status-${good ? 'good' : waiting ? 'waiting' : failed ? 'bad' : 'neutral'}`} aria-label={`Status: ${label}`}><Icon size={13} aria-hidden="true"/><span>{label}</span></span>
}

function statusLabel(value) {
  const labels = {
    registered: 'Not indexed', empty: 'Not indexed', indexed: 'Indexed', indexing: 'Indexing', failed: 'Failed',
    awaiting_approval: 'Awaiting approval', applying: 'Applying', testing: 'Testing', completed: 'Completed',
    tests_failed: 'Tests failed', rolled_back: 'Rolled back', rollback_failed: 'Rollback failed', rejected: 'Rejected',
  }
  return labels[value] || value.replaceAll('_', ' ')
}

function DiffView({ diff, path }) {
  return <pre className="diff-viewer" aria-label={`Unified diff for ${path}`}><code>{diff.split('\n').map((line, index) => {
    const kind = line.startsWith('+++') || line.startsWith('---') ? 'diff-header' : line.startsWith('+') ? 'diff-added' : line.startsWith('-') ? 'diff-removed' : ''
    return <span className={`diff-line ${kind}`} key={`${index}-${line}`}>{line || ' '}</span>
  })}</code></pre>
}

function Modal({ modal, onClose, onConfirm, busy }) {
  const ref = useRef(null)
  const wasOpen = useRef(false)
  const returnFocus = useRef(null)
  useEffect(() => {
    if (!modal) {
      if (wasOpen.current) {
        if (returnFocus.current?.isConnected) returnFocus.current.focus()
        else document.querySelector('.page-heading h1')?.focus()
        returnFocus.current = null
      }
      wasOpen.current = false
      return
    }
    if (!wasOpen.current) returnFocus.current = document.activeElement
    wasOpen.current = true
    ref.current?.querySelector('button')?.focus()
    const onKeyDown = event => {
      if (event.key === 'Escape' && !busy) onClose()
      if (event.key === 'Tab') {
        const nodes = [...ref.current.querySelectorAll('button:not(:disabled)')]
        if (!nodes.length) return
        const first = nodes[0], last = nodes[nodes.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [modal, busy, onClose])
  if (!modal) return null
  return <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose() }}>
    <section className="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title" aria-describedby="modal-description" ref={ref}>
      <button className="icon-button modal-close" aria-label="Close dialog" onClick={onClose} disabled={busy}><X size={16}/></button>
      <div className={`modal-symbol ${modal.danger ? 'symbol-danger' : ''}`}>{modal.danger ? <Trash2 size={20}/> : modal.rollback ? <RotateCcw size={20}/> : <ShieldCheck size={20}/>}</div>
      <h2 id="modal-title">{modal.title}</h2>
      <p id="modal-description">{modal.description}</p>
      {modal.files?.length > 0 && <div className="modal-files"><span>FILES AFFECTED</span>{modal.files.map(file => <code key={file}>{file}</code>)}</div>}
      {modal.note && <div className="modal-note">{modal.note}</div>}
      <div className="modal-actions">
        <button className="button button-secondary" onClick={onClose} disabled={busy}>Cancel</button>
        <button className={`button ${modal.danger ? 'button-danger' : 'button-primary'}`} onClick={onConfirm} disabled={busy}>
          {busy ? <LoaderCircle className="spin" size={15}/> : null}{busy ? modal.busyLabel : modal.confirmLabel}
        </button>
      </div>
    </section>
  </div>
}

function App() {
  const [projects, setProjects] = useState([])
  const [projectId, setProjectId] = useState('')
  const [tasks, setTasks] = useState([])
  const [task, setTask] = useState(null)
  const [question, setQuestion] = useState('')
  const [taskRequest, setTaskRequest] = useState('')
  const [answer, setAnswer] = useState(null)
  const [taskMode, setTaskMode] = useState('auto')
  const [workspaceTab, setWorkspaceTab] = useState('codebase')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [toast, setToast] = useState('')
  const [git, setGit] = useState(null)
  const [projectPath, setProjectPath] = useState('')
  const [modal, setModal] = useState(null)
  const [mobileNav, setMobileNav] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [expandedSources, setExpandedSources] = useState({})
  const closeModal = useCallback(() => setModal(null), [])
  const [isNarrow, setIsNarrow] = useState(() => window.matchMedia('(max-width: 720px)').matches)
  const sidebarRef = useRef(null)
  const mobileTriggerRef = useRef(null)
  const mobileNavWasOpen = useRef(false)

  const selected = projects.find(project => project.id === projectId)
  const refresh = useCallback(async () => {
    try {
      const projectData = await api('/projects')
      setProjects(projectData.projects)
      setBackendConnected(true)
      if (projectId) {
        const history = await api(`/tasks?project_id=${encodeURIComponent(projectId)}`)
        setTasks(history.tasks)
      } else setTasks([])
    } catch (cause) {
      setBackendConnected(false)
      setError(cause.message)
    }
  }, [projectId])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    if (!selected || selected.status !== 'indexing') return
    const timer = setInterval(() => { api(`/projects/${encodeURIComponent(selected.id)}`).then(latest => setProjects(current => current.map(item => item.id === latest.id ? latest : item))).catch(() => {}) }, 1000)
    return () => clearInterval(timer)
  }, [selected?.id, selected?.status])
  useEffect(() => {
    const media = window.matchMedia('(max-width: 720px)')
    const update = event => setIsNarrow(event.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    if (mobileNavWasOpen.current && !mobileNav && isNarrow) mobileTriggerRef.current?.focus()
    mobileNavWasOpen.current = mobileNav
  }, [mobileNav, isNarrow])
  useEffect(() => {
    if (!mobileNav) return
    sidebarRef.current?.querySelector('.nav-close')?.focus()
    const onKeyDown = event => {
      if (event.key === 'Escape') { setMobileNav(false); mobileTriggerRef.current?.focus() }
      if (event.key === 'Tab') {
        const nodes = [...sidebarRef.current.querySelectorAll('a[href],button:not(:disabled),input:not(:disabled),select:not(:disabled)')]
        if (!nodes.length) return
        const first = nodes[0], last = nodes[nodes.length - 1]
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [mobileNav])
  useEffect(() => {
    if (!projectId) return
    let active = true
    api(`/projects/${encodeURIComponent(projectId)}/git`).then(result => { if (active) setGit(result) }).catch(() => { if (active) setGit(null) })
    return () => { active = false }
  }, [projectId])
  useEffect(() => {
    if (!task?.task_id || terminal.has(task.status) || (task.status === 'awaiting_approval' && !busy)) return
    const timer = setInterval(async () => {
      try {
        const latest = await api(`/tasks/${task.task_id}`)
        setTask(latest)
        await refresh()
      } catch (cause) { setError(cause.message) }
    }, 1000)
    return () => clearInterval(timer)
  }, [task?.task_id, task?.status, refresh, busy])
  useEffect(() => {
    if (!toast) return
    const timer = setTimeout(() => setToast(''), 3200)
    return () => clearTimeout(timer)
  }, [toast])

  async function perform(label, action, successMessage) {
    setError('')
    setBusy(label)
    try {
      const result = await action()
      if (successMessage) setToast(successMessage)
      return result
    } catch (cause) {
      setError(cause.message || 'The action could not be completed. Try again.')
      return null
    } finally { setBusy('') }
  }

  async function runProjectIndex(path) {
    const project = await api('/projects/index?background=true', { method: 'POST', body: JSON.stringify({ project_path: path }) })
    if (project.id !== projectId) { setTask(null); setAnswer(null); setGit(null); setTasks([]) }
    setProjectId(project.id)
    setProjects(current => current.some(item => item.id === project.id)
      ? current.map(item => item.id === project.id ? project : item)
      : [project, ...current])
    let latest = project
    const deadline = Date.now() + 30 * 60 * 1000
    while (latest.status === 'indexing' && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      latest = await api(`/projects/${encodeURIComponent(project.id)}`)
      setProjects(current => current.map(item => item.id === latest.id ? latest : item))
    }
    if (latest.status === 'indexing') throw new Error('Indexing is taking longer than expected. Check the project progress and backend logs before retrying.')
    if (latest.status !== 'indexed') throw new Error(latest.indexing_error || 'Indexing did not complete. Check Ollama and retry.')
    await refresh()
    return latest
  }

  async function indexProject(event) {
    event.preventDefault()
    const path = projectPath.trim()
    const result = await perform('index', async () => {
      const project = await runProjectIndex(path)
      setTask(null); setAnswer(null); setGit(null); setProjectPath('')
      return project
    }, 'Project indexed successfully')
    if (!result && !error) setError('Indexing could not complete. Check the project path and backend status, then retry.')
  }

  async function reindexProject() {
    if (!selected?.path) return
    await perform('index', () => runProjectIndex(selected.path), 'Project reindexed successfully')
  }

  async function askQuestion(event) {
    event.preventDefault()
    if (!projectId || !question.trim()) return
    const result = await perform('ask', async () => api('/ask', {
      method: 'POST', body: JSON.stringify({ project_id: projectId, question: question.trim() }),
    }), 'Answer ready')
    if (result) { setAnswer(result); setQuestion(''); setExpandedSources({}) }
  }

  async function startTask(event) {
    event.preventDefault()
    if (!projectId || !taskRequest.trim()) return
    const result = await perform('start-task', async () => api('/tasks?background=true', {
      method: 'POST', body: JSON.stringify({ project_id: projectId, request: taskRequest.trim(), mode: taskMode }),
    }), 'Task started')
    if (result) { setTask(result); setTasks(current => [result, ...current]); setTaskRequest('') }
  }

  function selectProject(id) {
    setProjectId(id); setTask(null); setAnswer(null); setGit(null); setTasks([]); setError(''); setMobileNav(false); setWorkspaceTab('codebase')
  }

  async function removeProject() {
    const projectName = selected?.name || 'Project'
    const removedId = projectId
    const result = await perform('remove', async () => api(`/projects/${encodeURIComponent(projectId)}`, { method: 'DELETE' }), `${projectName} removed`)
    setModal(null)
    if (result) {
      setProjects(current => current.filter(project => project.id !== removedId))
      setProjectId(''); setTasks([]); setTask(null); setAnswer(null); setGit(null)
      try {
        const projectData = await api('/projects')
        setProjects(projectData.projects)
        setBackendConnected(true)
      } catch (cause) { setError(cause.message) }
    }
  }

  async function inspectGit() {
    const result = await perform('git', async () => api(`/projects/${encodeURIComponent(projectId)}/git`))
    if (result) setGit(result)
  }

  async function executeTaskAction(action) {
    if (!task?.task_id) return
    const messages = { approve: 'Approval submitted', reject: 'Proposal rejected', rollback: 'Rollback completed' }
    const result = await perform(action, async () => api(`/tasks/${task.task_id}/${action}`, { method: 'POST' }), messages[action])
    setModal(null)
    if (result) { setTask(result); await refresh() }
  }

  const activeTask = task || tasks[0]

  return <div className="app-shell">
    <div className={`mobile-scrim ${mobileNav ? 'visible' : ''}`} onClick={() => setMobileNav(false)} aria-hidden="true"/>
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`} aria-label="Workspace navigation" ref={sidebarRef} inert={isNarrow && !mobileNav}>
      <a className="brand" href="#workspace" aria-label="Buildmate workspace">
        <span className="brand-mark"><Code2 size={18} strokeWidth={2.2}/></span>
        <span className="brand-copy"><strong>Buildmate</strong><small>AI SOFTWARE ENGINEERING AGENT</small></span>
      </a>
      <button className="icon-button nav-close" aria-label="Close workspace navigation" onClick={() => setMobileNav(false)}><X size={17}/></button>

      <div className="sidebar-section-label">WORKSPACE</div>
      <label className="field-label" htmlFor="project-select">Project</label>
      <div className="select-wrap"><select id="project-select" value={projectId} onChange={event => selectProject(event.target.value)} aria-label="Select a registered project">
        <option value="">Select project</option>
        {projects.map(project => <option key={project.id} value={project.id}>{project.name} · {statusLabel(project.status)} · {project.file_count} files · {project.chunk_count} chunks</option>)}
      </select><ChevronDown size={15} aria-hidden="true"/></div>
      {selected && <div className="selected-project" aria-live="polite">
        <div className="selected-project-title"><strong>{selected.name}</strong><Status value={selected.status}>{statusLabel(selected.status)}</Status></div>
        <div className="project-counts"><span><b>{selected.file_count}</b> files</span><i>·</i><span><b>{selected.chunk_count}</b> chunks</span></div>
      </div>}

      <div className="sidebar-divider"/>
      <div className="sidebar-section-label project-label">PROJECT</div>
      <form className="index-form" onSubmit={indexProject}>
        <label className="field-label" htmlFor="project-path">Register / index project</label>
        <input id="project-path" value={projectPath} onChange={event => setProjectPath(event.target.value)} placeholder="Local project path" autoComplete="off" required aria-describedby="path-help"/>
        <span id="path-help" className="sr-only">Enter a path accessible from the backend host.</span>
        <button className="button button-secondary index-button" disabled={!!busy || !projectPath.trim()}>
          {busy === 'index' ? <><LoaderCircle className="spin" size={14}/>Indexing project…</> : <><Plus size={14}/>Register project</>}
        </button>
      </form>
      {selected?.status === 'indexing' && <div className="index-progress" role="status" aria-live="polite"><LoaderCircle className="spin" size={14}/><span><strong>{selected.indexing_progress?.stage || 'Indexing'}</strong><small>{selected.indexing_progress?.files_processed || 0} / {selected.indexing_progress?.files_discovered || 0} files · {selected.indexing_progress?.chunks_created || 0} chunks · {selected.indexing_progress?.embedding_batches_completed || 0} / {selected.indexing_progress?.total_embedding_batches || 0} embedding batches</small></span></div>}
      {selected?.status === 'failed' && selected.indexing_error && <div className="index-error" role="status"><AlertCircle size={14}/><span>{selected.indexing_error} Use Reindex to retry.</span></div>}

      <div className="history-heading"><span className="sidebar-section-label">RECENT TASKS</span><History size={14} aria-hidden="true"/></div>
      <nav className="history-list" aria-label="Recent tasks">
        {tasks.map(item => <button key={item.task_id} className={`history-item ${activeTask?.task_id === item.task_id ? 'history-selected' : ''}`} onClick={() => { setTask(item); setWorkspaceTab('agent'); setMobileNav(false) }}>
          <span className="history-indicator" aria-hidden="true"/><span className="history-content"><span className="history-title">{item.request}</span><span className="history-meta"><Status value={item.status}/><time dateTime={item.updated_at}>{relativeTime(item.updated_at)}</time></span></span>
        </button>)}
        {!tasks.length && <div className="history-empty"><strong>No tasks yet</strong><span>Start an investigation to see agent activity here.</span></div>}
      </nav>

      <div className="sidebar-bottom"><span className={`connection-dot ${backendConnected ? 'is-connected' : ''}`} aria-hidden="true"/><span>{backendConnected ? 'Backend connected' : 'Connecting to backend'}</span><span className="local-agent-label">Local agent</span></div>
    </aside>

    <main className="workspace" id="workspace">
      <header className="page-header">
        <button ref={mobileTriggerRef} className="icon-button mobile-menu" aria-label="Open workspace navigation" aria-expanded={mobileNav} onClick={() => setMobileNav(true)}><Menu size={19}/></button>
        <div className="page-heading"><div className="eyebrow">ENGINEERING WORKSPACE</div><h1 tabIndex={-1}>{selected?.name || 'Your codebase, understood'}</h1></div>
        {selected && <div className="project-header-meta"><Status value={selected.status}>{statusLabel(selected.status)}</Status><span>{selected.file_count} files</span><i>·</i><span>{selected.chunk_count} chunks</span>{git && <Status value={git.changed_files?.length ? 'warning' : 'passed'}>{git.changed_files?.length ? `${git.changed_files.length} changed` : 'Git clean'}</Status>}</div>}
        {selected && <button className="button button-secondary reindex-button" onClick={reindexProject} disabled={!!busy} aria-label={`Reindex ${selected.name}`}>{busy === 'index' ? <><LoaderCircle className="spin" size={14}/>Reindexing…</> : <><RotateCcw size={14}/>Reindex</>}</button>}
        {selected && <button className="remove-project-button" onClick={() => setModal({
          danger: true, title: 'Remove from Buildmate?',
          description: 'This removes the project from Buildmate and deletes its index. Your source files will not be deleted.',
          note: 'Task history and verified rollback backups are retained.',
          confirmLabel: 'Remove from Buildmate', busyLabel: 'Removing…',
        })}><Trash2 size={14}/>Remove from Buildmate</button>}
      </header>

      {error && <div className="notice notice-error" role="alert"><AlertCircle size={16}/><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><X size={15}/></button></div>}

      {!selected ? <section className="welcome-panel" aria-labelledby="welcome-title">
        <div className="welcome-mark"><Code2 size={23}/></div><span className="eyebrow">UNDERSTAND · INVESTIGATE · VERIFY</span>
        <h2 id="welcome-title">Start with a project</h2>
        <p>Register a local codebase to ask questions, investigate issues, and review proposed fixes.</p>
        <form className="welcome-form" onSubmit={indexProject}><label className="sr-only" htmlFor="welcome-path">Project path</label><input id="welcome-path" value={projectPath} onChange={event => setProjectPath(event.target.value)} placeholder="Project path accessible to the backend" required/><button className="button button-primary" disabled={!!busy || !projectPath.trim()}>{busy === 'index' ? <><LoaderCircle className="spin" size={15}/>Indexing…</> : <>Register project<ArrowUpRight size={15}/></>}</button></form>
        {busy === 'index' && <p className="progress-copy" role="status">Indexing project files and preparing search…</p>}
      </section> : <>
        <section className="workspace-intro">
          <div className="workspace-overview"><div className="intro-kicker"><span className={`connection-dot ${selected.status === 'indexed' ? 'is-connected' : ''}`}/> {statusLabel(selected.status).toUpperCase()}</div><h2>{workspaceTab === 'codebase' ? 'Understand your codebase' : 'Run an engineering task'}</h2><p>{workspaceTab === 'codebase' ? 'Ask a question and trace the answer back to source.' : 'Investigate, review, and propose changes for approval.'}</p><code className="project-location" title={selected.path}>{selected.path}</code></div>
          <div className="workspace-switch" role="tablist" aria-label="Project workspace" onKeyDown={event => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
            event.preventDefault()
            const next = event.key === 'Home' ? 'codebase' : event.key === 'End' ? 'agent' : event.key === 'ArrowRight' ? 'agent' : 'codebase'
            setWorkspaceTab(next)
            document.getElementById(next === 'codebase' ? 'tab-codebase' : 'tab-agent')?.focus()
          }}>
            <button id="tab-codebase" role="tab" tabIndex={workspaceTab === 'codebase' ? 0 : -1} aria-selected={workspaceTab === 'codebase'} aria-controls="workspace-panel" className={workspaceTab === 'codebase' ? 'active' : ''} onClick={() => setWorkspaceTab('codebase')}><MessageSquareText size={15}/>Codebase</button>
            <button id="tab-agent" role="tab" tabIndex={workspaceTab === 'agent' ? 0 : -1} aria-selected={workspaceTab === 'agent'} aria-controls="workspace-panel" className={workspaceTab === 'agent' ? 'active' : ''} onClick={() => setWorkspaceTab('agent')}><Sparkles size={15}/>Agent{tasks.length > 0 && <span className="tab-count">{tasks.length}</span>}</button>
          </div>
        </section>

        <div className={`content-grid workspace-${workspaceTab}`} id="workspace-panel" role="tabpanel" aria-labelledby={workspaceTab === 'codebase' ? 'tab-codebase' : 'tab-agent'}>
          <div className="primary-column">
            {workspaceTab === 'codebase' && <section className="section-block question-section" aria-labelledby="question-title">
              <div className="section-heading"><div className="section-icon icon-violet"><MessageSquareText size={17}/></div><div><h2 id="question-title">Ask your codebase</h2><p>Get grounded answers from your indexed source code.</p></div></div>
              <form onSubmit={askQuestion}>
                <label className="sr-only" htmlFor="code-question">Question about the codebase</label>
                <textarea id="code-question" className="question-input" value={question} onChange={event => setQuestion(event.target.value)} placeholder="Where is authentication handled?" rows={3} required disabled={!!busy}/>
                <div className="input-actions"><span className="input-hint">Answers include file and line references.</span><button className="button button-primary" disabled={!!busy || !question.trim()}>{busy === 'ask' ? <><LoaderCircle className="spin" size={15}/>Searching codebase…</> : <><Search size={15}/>Ask question</>}</button></div>
              </form>
              {busy === 'ask' && <div className="progress-line" role="status"><span className="progress-dot"/>Retrieving relevant context and generating an answer…</div>}
              {answer && <div className="answer-panel" aria-live="polite"><div className="answer-title"><CheckCircle2 size={15}/><strong>Answer</strong></div><p className="answer-text">{answer.answer}</p><div className="sources-heading"><span>Sources</span><span>{answer.sources?.length || 0} references</span></div>{answer.sources?.length ? <div className="source-list">{answer.sources.map((source, index) => {
                const expanded = !!expandedSources[index]
                return <details className="source-row" key={`${source.file}-${source.start_line}-${index}`} open={expanded} onToggle={event => setExpandedSources(current => ({ ...current, [index]: event.currentTarget.open }))}>
                  <summary><FileCode2 size={15}/><code>{source.file}</code><span>{source.start_line}–{source.end_line}</span><ChevronRight className="source-chevron" size={14}/></summary>
                  <div className="source-details"><strong>{source.symbol || 'Source excerpt'}</strong><span>{source.type || 'code'}{source.parent ? ` · ${source.parent}` : ''}</span></div>
                </details>
              })}</div> : <div className="inline-empty"><strong>No relevant source found</strong><span>Try a more specific question.</span></div>}</div>}
            </section>}

            {workspaceTab === 'agent' && <>
            <section className="section-block task-section" aria-labelledby="task-title">
              <div className="section-heading"><div className="section-icon icon-blue"><Sparkles size={17}/></div><div><h2 id="task-title">Agent task</h2><p>Investigate, review, or propose an approval-gated fix.</p></div></div>
              <form onSubmit={startTask}>
                <label className="sr-only" htmlFor="task-request">Describe the agent task</label>
                <textarea id="task-request" className="task-input" value={taskRequest} onChange={event => setTaskRequest(event.target.value)} placeholder="Describe what you want the agent to investigate…" rows={3} required disabled={!!busy}/>
                <div className="input-actions"><label className="sr-only" htmlFor="task-mode">Task mode</label><select id="task-mode" className="mode-select" value={taskMode} onChange={event => setTaskMode(event.target.value)} disabled={!!busy}><option value="auto">Automatic</option><option value="investigate">Investigate</option><option value="review">Review</option><option value="change">Propose fix</option></select><button className="button button-primary" disabled={!!busy || !taskRequest.trim()}>{busy === 'start-task' ? <><LoaderCircle className="spin" size={15}/>Starting task…</> : <><Sparkles size={15}/>Start task</>}</button></div>
              </form>
              {busy === 'start-task' && <div className="progress-line" role="status"><span className="progress-dot"/>Starting agent investigation…</div>}
            </section>

            {activeTask && <TaskResult task={activeTask} busy={busy} onAction={action => {
              if (action === 'reject') { executeTaskAction(action); return }
              const changes = activeTask.proposed_changes || []
              setModal(action === 'rollback' ? {
                rollback: true, title: 'Roll back these changes?',
                description: 'Buildmate will restore the verified backups for these files. Files changed manually since approval will be left untouched and the rollback will be refused.',
                files: changes.map(change => change.path), confirmLabel: 'Roll back changes', busyLabel: 'Rolling back…',
              } : {
                title: 'Approve and apply changes?',
                description: 'Review changes before applying. Buildmate will update the listed files, create backups, and run the registered test suite.',
                files: changes.map(change => change.path), note: 'Changes are limited to the selected project. Tests run after the approved diff is applied.',
                confirmLabel: 'Approve & apply', busyLabel: 'Applying changes…',
              })
            }}/ >}
            </>}
          </div>

          <aside className="secondary-column" aria-label="Project and agent details">
            {workspaceTab === 'agent' && <section className="side-panel activity-panel" aria-labelledby="activity-title"><div className="panel-heading"><div><h2 id="activity-title">Agent activity</h2><p>{activeTask ? `Task · ${statusLabel(activeTask.status)}` : 'Tool events and task progress'}</p></div><Activity size={16} aria-hidden="true"/></div>
              {activeTask ? <ActivityTimeline task={activeTask}/> : <div className="panel-empty"><Activity size={18}/><strong>No activity yet</strong><span>Start a task to see the investigation here.</span></div>}
            </section>}
            <section className="side-panel git-panel" aria-labelledby="git-title"><div className="panel-heading"><div><h2 id="git-title">Git status</h2><p>Read-only repository details</p></div><GitBranch size={16} aria-hidden="true"/></div>
              <button className="button button-secondary git-refresh" onClick={inspectGit} disabled={!!busy}>{busy === 'git' ? <><LoaderCircle className="spin" size={14}/>Checking repository…</> : <><GitBranch size={14}/>Inspect repository</>}</button>
              {git ? git.is_git_repository ? <div className="git-summary"><div className="git-branch"><GitBranch size={14}/><code>{git.branch || 'Detached HEAD'}</code><Status value={git.changed_files?.length ? 'warning' : 'passed'}>{git.changed_files?.length ? `${git.changed_files.length} changed` : 'Working tree clean'}</Status></div>{git.changed_files?.length > 0 && <details className="git-files"><summary>Changed files <ChevronDown size={13}/></summary><ul>{git.changed_files.map((file, index) => <li key={`${file.path}-${index}`}><span>{file.status}</span><code>{file.path}</code></li>)}</ul></details>}{git.diff && <details className="git-diff-details"><summary>Review diff <ChevronDown size={13}/></summary><DiffView diff={`${git.diff}${git.diff_truncated ? '\n… diff truncated' : ''}`} path="Git working tree"/></details>}</div> : <div className="panel-empty"><GitBranch size={18}/><strong>Git repository not detected</strong><span>Git status appears when this project contains a repository.</span></div> : <p className="git-placeholder">Inspect the repository to view its branch and working tree.</p>}
            </section>
            {workspaceTab === 'agent' && <div className="trust-note"><ShieldCheck size={15}/><span>Code changes are shown for review before anything is applied.</span></div>}
          </aside>
        </div>
      </>}
      <footer className="page-footer"><span>Buildmate</span><i>·</i><span>Understand. Investigate. Verify.</span></footer>
    </main>

    {toast && <div className="toast" role="status" aria-live="polite"><CheckCircle2 size={16}/><span>{toast}</span><button className="icon-button" aria-label="Dismiss notification" onClick={() => setToast('')}><X size={14}/></button></div>}
    <Modal modal={modal} busy={!!busy} onClose={closeModal} onConfirm={() => modal?.danger ? removeProject() : executeTaskAction(modal?.rollback ? 'rollback' : 'approve')}/>
  </div>
}

function ActivityTimeline({ task }) {
  const toolLabel = { search_code: 'Searching code', read_file: 'Reading source files', find_symbol: 'Finding symbol', inspect_project: 'Inspecting project', run_tests: 'Running tests' }
  const events = [
    ...(task.actions || []).filter(action => action.action !== 'tool_call').map((action, index) => ({ id: `a-${index}`, type: action.action, label: ({ investigation_started: 'Agent started', investigated: 'Investigation complete', change_proposed: 'Changes proposed', approved: 'Changes approved', tests_passed: 'Tests passed', tests_failed: 'Tests failed', fix_proposed: 'Follow-up change proposed', rejected: 'Changes rejected', rolled_back: 'Rollback completed', apply_failed: 'Apply failed' })[action.action] || action.action.replaceAll('_', ' '), detail: action.summary, time: action.at })),
    ...(task.tool_calls || []).map((call, index) => ({ id: `t-${index}`, type: call.tool, label: toolLabel[call.tool] || 'Using registered tool', detail: call.arguments?.file_path || call.arguments?.symbol || (call.success ? 'Completed' : 'Could not complete'), time: call.at, duration: call.elapsed_ms })),
  ].sort((first, second) => (first.time || '').localeCompare(second.time || '') || first.id.localeCompare(second.id))
  if (['investigating', 'generating_proposal', 'testing', 'analyzing_test_failure', 'rolling_back'].includes(task.status)) events.push({ id: 'active', type: 'active', label: ({ investigating: 'Investigating request', generating_proposal: 'Preparing proposed changes', testing: 'Running tests', analyzing_test_failure: 'Reviewing test failure', rolling_back: 'Rolling back changes' })[task.status], detail: 'In progress' })
  if (task.status === 'awaiting_approval') events.push({ id: 'approval', type: 'awaiting_approval', label: 'Waiting for approval', detail: 'Review the proposed diff before applying.' })
  if (!events.length) return <div className="panel-empty"><Activity size={18}/><strong>No activity yet</strong><span>Task events will appear here.</span></div>
  return <ol className="timeline">{events.map((event, index) => {
    const isActive = event.type === 'active'
    const isBad = ['failed', 'tests_failed', 'apply_failed', 'rollback_failed'].includes(event.type)
    const Icon = isBad ? AlertCircle : isActive || event.type === 'awaiting_approval' ? Clock3 : event.type.includes('tool') || toolLabel[event.type] ? ChevronRight : Check
    return <li key={event.id} className={`timeline-event ${isActive ? 'event-active' : ''} ${isBad ? 'event-failed' : ''}`}>
      <span className="timeline-icon"><Icon size={13}/></span><div className="timeline-copy"><strong>{event.label}</strong><span>{event.detail}</span><small>{event.time ? relativeTime(event.time) : event.duration ? `${event.duration} ms` : isActive ? 'In progress' : ''}</small></div>
    </li>
  })}</ol>
}

function TaskResult({ task, busy, onAction }) {
  const pendingChanges = task.proposed_changes?.filter(change => change.approval_status === 'pending') || []
  const canApprove = task.status === 'awaiting_approval' && pendingChanges.length > 0
  const canRollback = task.applications?.length > 0 && !['rolled_back', 'rolling_back'].includes(task.status)
  const counts = (task.proposed_changes || []).reduce((result, change) => {
    for (const line of change.diff.split('\n')) {
      if (line.startsWith('+') && !line.startsWith('+++')) result.added++
      if (line.startsWith('-') && !line.startsWith('---')) result.removed++
    }
    return result
  }, { added: 0, removed: 0 })
  return <section className="section-block result-section" aria-labelledby="result-title"><div className="section-heading"><div className="section-icon icon-amber"><FileCode2 size={17}/></div><div><h2 id="result-title">Task result</h2><p><Status value={task.status}>{statusLabel(task.status)}</Status></p></div></div>
    {task.request && <div className="task-request"><strong>Your request</strong><p>{task.request}</p></div>}
    {task.answer && <div className="task-answer"><strong>{task.mode === 'review' ? 'Review findings' : task.mode === 'investigate' ? 'Investigation' : 'Agent summary'}</strong><p>{task.answer}</p></div>}
    {task.diagnosis && task.diagnosis !== task.answer && <div className="task-answer"><strong>Diagnosis</strong><p>{task.diagnosis}</p></div>}
    {task.proposed_changes?.length > 0 && <div className="changes-review"><div className="changes-header"><div><strong>Proposed changes</strong><span>{task.proposed_changes.length} {task.proposed_changes.length === 1 ? 'file' : 'files'} changed</span></div><div className="diff-counts"><span className="added-count"><Plus size={12}/>{counts.added}</span><span className="removed-count"><Minus size={12}/>{counts.removed}</span></div></div>
      {task.proposed_changes.map((change, index) => <article className="change-file" key={`${change.path}-${index}`}><header className="change-file-header"><div><FileCode2 size={14}/><code>{change.path}</code></div><Status value={change.approval_status}>{change.approval_status}</Status></header><p className="change-reason">{change.reason}</p><DiffView diff={change.diff} path={change.path}/></article>)}
      {canApprove && <div className="approval-callout"><ShieldCheck size={15}/><span>Review changes before applying. Your approval is required to modify project files.</span></div>}
    </div>}
    {task.status === 'investigating' && <div className="progress-line" role="status"><LoaderCircle className="spin" size={14}/>Agent investigating the selected project…</div>}
    {task.status === 'generating_proposal' && <div className="progress-line" role="status"><LoaderCircle className="spin" size={14}/>Preparing proposed changes…</div>}
    {task.status === 'testing' && <div className="progress-line" role="status"><LoaderCircle className="spin" size={14}/>Applying approved changes and running tests…</div>}
    {task.test_results?.map((result, index) => {
      const couldNotRun = !result.success || result.exit_code == null
      const testLabel = result.passed ? 'Tests passed' : couldNotRun ? 'Tests could not run' : 'Tests failed'
      const failedCount = result.failed_count ?? (couldNotRun || result.passed ? 0 : 1)
      return <div className={`test-summary ${result.passed ? 'test-passed' : couldNotRun ? '' : 'test-failed'}`} key={`${result.attempt}-${index}`}><div><Status value={result.passed ? 'passed' : couldNotRun ? 'unknown' : 'failed'}>{testLabel}</Status><span>Attempt {result.attempt} · {result.passed_count ?? 0} passed · {failedCount} failed · {result.skipped_count ?? 0} skipped · {result.duration_seconds ?? '?'} sec</span></div>{!result.passed && <p>{couldNotRun ? 'Changes were applied, but Buildmate could not run the registered tests.' : 'Changes were applied, but the registered tests failed. Review the output before approving any follow-up proposal.'}</p>}<details><summary>Test output <ChevronDown size={13}/></summary><pre>{result.output || 'No test output was returned.'}</pre></details></div>
    })}
    {task.error?.message ? <div className="notice notice-error task-error" role="alert"><AlertCircle size={15} aria-hidden="true"/><span><strong>{task.error.stage === 'proposal' ? 'Proposal generation failed' : 'Investigation failed'}:</strong> {task.error.message}{task.error.cause_message && <small>Diagnostic: {task.error.cause_message}</small>}</span></div> : task.result && <p className="final-result">{task.result}</p>}
    <div className="task-actions">{canApprove && <><button className="button button-secondary" disabled={!!busy} onClick={() => onAction('reject')}><X size={14}/>Reject</button><button className="button button-primary" disabled={!!busy} onClick={() => onAction('approve')}><Check size={14}/>Approve changes</button></>}{canRollback && <button className="button button-secondary" disabled={!!busy} onClick={() => onAction('rollback')}><RotateCcw size={14}/>Roll back</button>}{busy === 'approve' && <span className="action-progress" role="status"><LoaderCircle className="spin" size={14}/>Applying changes and running tests…</span>}{busy === 'rollback' && <span className="action-progress" role="status"><LoaderCircle className="spin" size={14}/>Restoring verified backups…</span>}</div>
  </section>
}

createRoot(document.getElementById('root')).render(<App />)
