from features.multi_task.tasks import split_tasks
from features.multi_task.executor import run_parallel
from features.multi_task.agents import get_agent
def run_multi_task_project(user_input: str):
    """Manager - runs multiple tasks parallel"""
    print(f"?? Manager: {user_input}")
    tasks = split_tasks(user_input)
    results = run_parallel(tasks)
    final = get_agent("manager")(f"Combine these results into report: {results}")
    return results, final
