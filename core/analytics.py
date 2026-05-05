import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from core.config import CONFIG


class Analytics:
    """Enterprise analytics"""
    
    @staticmethod
    def _get_conn():
        """Get database connection - FIX for db.conn"""
        db_path = Path(CONFIG.DATABASE_URL.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return conn
    
    @staticmethod
    def log_interaction(user_id: int, session_id: str, model: str,
                        prompt_tokens: int, completion_tokens: int,
                        latency_ms: float, metadata: Dict = None):
        """Log interaction for analytics"""
        try:
            with Analytics._get_conn() as conn:
                conn.execute("""
                    INSERT INTO analytics 
                    (user_id, session_id, model, prompt_tokens, completion_tokens, 
                     total_tokens, latency_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    latency_ms,
                    json.dumps(metadata or {})
                ))
                conn.commit()
        except Exception as e:
            print(f"Analytics log error: {e}")
    
    @staticmethod
    def get_dashboard_data(user_id: int, days: int = 30) -> Dict:
        """Get analytics dashboard data"""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            with Analytics._get_conn() as conn:
                # Total stats
                stats = conn.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        SUM(total_tokens) as total_tokens,
                        AVG(latency_ms) as avg_latency
                    FROM analytics
                    WHERE user_id = ? AND timestamp > ?
                """, (user_id, since)).fetchone()
                
                # Daily usage
                daily = conn.execute("""
                    SELECT 
                        date(timestamp) as day,
                        COUNT(*) as interactions,
                        SUM(total_tokens) as tokens
                    FROM analytics
                    WHERE user_id = ? AND timestamp > ?
                    GROUP BY date(timestamp)
                    ORDER BY day
                """, (user_id, since)).fetchall()
                
                # Model usage
                models = conn.execute("""
                    SELECT model, COUNT(*) as uses
                    FROM analytics
                    WHERE user_id = ? AND timestamp > ?
                    GROUP BY model
                    ORDER BY uses DESC
                """, (user_id, since)).fetchall()
                
                return {
                    "stats": dict(stats) if stats else {},
                    "daily": [dict(row) for row in daily],
                    "models": [dict(row) for row in models]
                }
        except Exception as e:
            print(f"Analytics dashboard error: {e}")
            return {"stats": {}, "daily": [], "models": []}


# =============================================================================
# EXPORT & INTEGRATION
# =============================================================================


