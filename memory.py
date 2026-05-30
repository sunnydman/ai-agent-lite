import sqlite3


class MemoryManager:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            session_id, role, content, timestamp)''')
        self.conn.commit()

    def save(self, session_id, role, content):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memory_fts (session_id, role, content, timestamp) VALUES (?, ?, ?, datetime('now'))",
            (session_id, role, content))
        self.conn.commit()

    def search(self, session_id, query, limit=5):
        """基于 FTS5 的全文检索策略"""
        cursor = self.conn.cursor()
        fts_query = self._build_fts_query(query)
        try:
            cursor.execute("""
                SELECT role, content FROM memory_fts
                WHERE session_id = ? AND memory_fts MATCH ?
                ORDER BY rank LIMIT ?
            """, (session_id, fts_query, limit))
            return [{"role": r, "content": c} for r, c in cursor.fetchall()]
        except sqlite3.Error:
            cursor.execute("""
                SELECT role, content FROM memory_fts
                WHERE session_id = ? AND content LIKE ?
                ORDER BY timestamp DESC LIMIT ?
            """, (session_id, f"%{query}%", limit))
            return [{"role": r, "content": c} for r, c in cursor.fetchall()]

    def _build_fts_query(self, query):
        tokens = [token.strip() for token in query.replace('"', " ").split() if token.strip()]
        if not tokens:
            tokens = [query.strip()]
        return " OR ".join(f'"{token}"' for token in tokens if token)

    def get_recent(self, session_id, limit=10):
        cursor = self.conn.cursor()
        cursor.execute("SELECT role, content FROM memory_fts WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                       (session_id, limit))
        return [{"role": r, "content": c} for r, c in reversed(cursor.fetchall())]

    def clear(self, session_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memory_fts WHERE session_id = ?", (session_id,))
        self.conn.commit()
