-- 001_init.sql  : create initial schema
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
);

CREATE TABLE IF NOT EXISTS project_repos (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    repo_id TEXT,
    UNIQUE(project_id, repo_id)
);

CREATE TABLE IF NOT EXISTS repos (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
    root_path TEXT,
    remote_url TEXT,
    default_branch TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS packages (
    id TEXT PRIMARY KEY,
    name TEXT,
    repo_id TEXT,
    language TEXT,
    virtual_path TEXT,
    physical_path TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    package_id TEXT,
    path TEXT,
    file_hash TEXT,
    last_updated BIGINT,
);

CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    file_id TEXT,
    package_id TEXT,
    parent_node_id TEXT,
    name TEXT,
    body TEXT,
    header TEXT,
    kind TEXT,
    subtype TEXT,
    docstring TEXT,
    comment TEXT,
    start_line INTEGER,
    end_line INTEGER,
    start_byte INTEGER,
    end_byte INTEGER,
    embedding_code_vec FLOAT[1024],
    embedding_model TEXT,
    search_boost FLOAT DEFAULT 1.0,
    fts_needle TEXT
);

CREATE TABLE IF NOT EXISTS import_edges (
    id TEXT PRIMARY KEY,
    repo_id TEXT,
    from_package_id TEXT,
    from_file_id TEXT,
    to_package_physical_path TEXT,
    to_package_virtual_path TEXT,
    to_package_id TEXT,
    alias TEXT,
    dot BOOLEAN,
    external BOOLEAN,
    raw TEXT
);

-- Table with precomputed lowercased path and basename for cheap matching/scoring
CREATE TABLE IF NOT EXISTS files_search (
    file_id TEXT PRIMARY KEY,
    path_lc TEXT NOT NULL,
    basename_lc TEXT NOT NULL
);

-- Trigram presence index (one row per distinct trigram per file)
CREATE TABLE IF NOT EXISTS file_trigrams (
    file_id TEXT NOT NULL,
    trigram TEXT NOT NULL,
    PRIMARY KEY (file_id, trigram)
);

