-- 003_repo_last_scanned.sql : add last_scanned column to repos

ALTER TABLE repos ADD COLUMN IF NOT EXISTS last_scanned BIGINT;
