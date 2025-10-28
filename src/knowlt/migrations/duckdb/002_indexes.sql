-- 002_indexes.sql  : create initial indexes

-- Indexes
-- Lookup index for trigram -> file ids
CREATE INDEX IF NOT EXISTS idx_file_trigrams_trigram ON file_trigrams(trigram);

-- Create indexes
-- repos
CREATE UNIQUE INDEX IF NOT EXISTS idx_repos_id          ON repos(id);
CREATE        INDEX IF NOT EXISTS idx_repos_root_path   ON repos(root_path);

-- packages
CREATE UNIQUE INDEX IF NOT EXISTS idx_packages_id       ON packages(id);
CREATE        INDEX IF NOT EXISTS idx_packages_physpath ON packages(physical_path);

-- files
CREATE UNIQUE INDEX IF NOT EXISTS idx_files_id          ON files(id);
CREATE        INDEX IF NOT EXISTS idx_files_path        ON files(path);
CREATE        INDEX IF NOT EXISTS idx_files_repo_id     ON files(repo_id);
CREATE        INDEX IF NOT EXISTS idx_files_package_id  ON files(package_id);

-- symbols
CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_id          ON nodes(id);
CREATE        INDEX IF NOT EXISTS idx_nodes_file_id     ON nodes(file_id);

-- import_edges
CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_id          ON import_edges(id);
CREATE        INDEX IF NOT EXISTS idx_edges_from_pkg_id ON import_edges(from_package_id);

-- Create vector indexes
SET hnsw_enable_experimental_persistence = true;

CREATE INDEX IF NOT EXISTS idx_nodes_code_vec ON nodes USING HNSW (embedding_code_vec) WITH (metric = 'cosine');;

-- Create FTS index
PRAGMA create_fts_index('nodes', 'id', 'fts_needle');
