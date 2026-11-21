-- ============================================================
-- 数据库初始化 SQL
-- PostgreSQL + pgvector + tsvector (全文+向量双引擎)
-- ============================================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- 1. 知识库表
-- ============================================================
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    doc_type VARCHAR(50) NOT NULL DEFAULT 'general',  -- general | code | contract | tech_doc | legal
    bm25_k1 REAL NOT NULL DEFAULT 1.5,                -- BM25 词频饱和度（per-Collection）
    bm25_b REAL NOT NULL DEFAULT 0.75,                -- BM25 长度归一化
    embedding_model VARCHAR(200) DEFAULT 'BAAI/bge-base-zh-v1.5',
    chunk_size INT NOT NULL DEFAULT 512,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- 2. 文档表
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    kb_id INT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,                    -- pdf | docx | pptx | xlsx | md | image | code
    file_path VARCHAR(1000),                           -- MinIO 存储路径
    file_size_bytes BIGINT,
    page_count INT,
    parse_status VARCHAR(30) NOT NULL DEFAULT 'pending', -- pending | parsing | completed | failed
    parse_error TEXT,
    checksum VARCHAR(64),                              -- SHA-256，用于增量更新去重
    metadata JSONB DEFAULT '{}',                       -- {author, created_at, permissions, department, ...}
    chunk_count INT DEFAULT 0,
    vectorized_chunks INT DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_documents_kb ON documents(kb_id);
CREATE INDEX idx_documents_status ON documents(parse_status);
CREATE INDEX idx_documents_checksum ON documents(checksum);
CREATE INDEX idx_documents_metadata ON documents USING GIN(metadata);

-- ============================================================
-- 3. 文档块表（三级块结构：base / parent / summary）
-- ============================================================
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kb_id INT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    chunk_type VARCHAR(20) NOT NULL DEFAULT 'base',     -- base | parent | summary
    chunk_index INT NOT NULL,                           -- 在文档内的序号
    parent_chunk_id INT REFERENCES chunks(id),          -- 父块关联
    content TEXT NOT NULL,
    content_tokens INT,                                 -- token 数估算
    heading VARCHAR(500),                               -- 所在章节标题
    section_path VARCHAR(1000),                         -- 标题层级路径: "第三章 > 3.1 > 3.1.2"
    page_start INT,
    page_end INT,
    embedding VECTOR(1024),                             -- pgvector 向量字段（1024维 BGE-M3）
    embedding_model VARCHAR(200),
    sparse_vector TSVECTOR,                             -- PostgreSQL 全文检索
    metadata JSONB DEFAULT '{}',                        -- {has_table, has_image, has_code, language, ...}
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_kb ON chunks(kb_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX idx_chunks_type ON chunks(chunk_type);
-- pgvector IVF 索引（1000个聚类中心）
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
-- 全文检索索引
CREATE INDEX idx_chunks_fts ON chunks USING GIN(sparse_vector);
-- JSONB 索引
CREATE INDEX idx_chunks_metadata ON chunks USING GIN(metadata);

-- ============================================================
-- 4. 用户表（文档级权限控制）
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(30) NOT NULL DEFAULT 'viewer',         -- admin | editor | viewer
    department VARCHAR(100),
    permission_groups VARCHAR(50)[] NOT NULL DEFAULT '{}', -- 权限组标签
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 文档级权限关联
CREATE TABLE IF NOT EXISTS document_permissions (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    principal_type VARCHAR(20) NOT NULL,                -- user | group | department
    principal_value VARCHAR(200) NOT NULL,
    permission_level VARCHAR(20) NOT NULL DEFAULT 'read', -- read | annotate | manage
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(doc_id, principal_type, principal_value)
);

CREATE INDEX idx_doc_perms_doc ON document_permissions(doc_id);

-- ============================================================
-- 5. 查询日志表（线上反馈闭环）
-- ============================================================
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id INT REFERENCES users(id),
    kb_ids INT[] NOT NULL,
    original_query TEXT NOT NULL,
    rewritten_queries TEXT[],                           -- 查询重写历史
    intent_type VARCHAR(20),                            -- factual | conceptual | relational | procedural | mixed
    retrieval_route VARCHAR(30),                        -- bm25 | dense | graph | hybrid
    recall_count INT,                                   -- 召回候选数
    sufficiency_score REAL,                             -- 检索充分性打分
    retrieval_rounds INT DEFAULT 1,                     -- Agentic RAG 重试轮数
    top_chunk_ids INT[],
    generated_answer TEXT,
    citations INT[],
    hallucination_rate REAL,                            -- NLI模型检测的幻觉比例
    user_feedback VARCHAR(20),                          -- positive | negative | skip
    latency_ms JSONB DEFAULT '{}',                      -- 各阶段延迟明细
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_query_logs_user ON query_logs(user_id);
CREATE INDEX idx_query_logs_intent ON query_logs(intent_type);
CREATE INDEX idx_query_logs_feedback ON query_logs(user_feedback);
CREATE INDEX idx_query_logs_created ON query_logs(created_at);

-- ============================================================
-- 6. 评估结果表
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    evaluator_type VARCHAR(30) NOT NULL,                -- ragas | custom_benchmark | chunk_quality | regression
    baseline_version VARCHAR(50),                       -- 对比基线版本
    target_version VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}',                 -- {context_precision, context_recall, faithfulness, ...}
    regression_cases JSONB DEFAULT '[]',                 -- 退化case列表
    improvement_cases JSONB DEFAULT '[]',                -- 改进case列表
    total_qa_pairs INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eval_run ON evaluation_results(run_id);
CREATE INDEX idx_eval_version ON evaluation_results(target_version);

-- ============================================================
-- 全文检索触发器：content 变更自动更新 tsvector
-- ============================================================
CREATE OR REPLACE FUNCTION update_chunk_fts() RETURNS TRIGGER AS $$
BEGIN
    NEW.sparse_vector := to_tsvector('simple', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_chunk_fts
    BEFORE INSERT OR UPDATE OF content ON chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();
