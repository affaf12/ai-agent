from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3, uuid, time
from datetime import datetime
from features.multi_task.agents import get_agent
def run_parallel(tasks, max_workers=5):
    """Parallel execution engine"""
    results = []
    print(f"? Running {len(tasks)} tasks...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_one, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
                print(f"? {futures[future]['id']} done")
            except Exception as e:
                print(f"? Error: {e}")
    return sorted(results, key=lambda x: x.get('priority', 0))
def run_one(task):
    """Execute single task"""
    conn = sqlite3.connect("rag_system.db")
    task_id = str(uuid.uuid4())
    # Log start
    conn.execute("""
        INSERT INTO agent_runs (id, agent_name, status, input, started_at)
        VALUES (?,?,?,?,?)
    """, (task_id, task["agent_type"], "running", task["description"], datetime.now()))
    conn.commit()
    try:
        start = time.time()
        result = get_agent(task["agent_type"])(task["description"])
        # Log success
        conn.execute("""
            UPDATE agent_runs SET status=?, output=?, finished_at=? WHERE id=?
        """, ("completed", result[:4000], datetime.now(), task_id))
        conn.commit()
        return {
            "id": task["id"],
            "task": task["description"],
            "agent": task["agent_type"],
            "result": result,
            "priority": task["priority"],
            "time": round(time.time() - start, 2)
        }
    finally:
        conn.close()
