def split_tasks(user_input: str):
    """Split project into 5 tasks"""
    return [
        {"id":"1","description":f"{user_input} - Research","agent_type":"researcher","priority":1},
        {"id":"2","description":f"{user_input} - Design","agent_type":"designer","priority":2},
        {"id":"3","description":f"{user_input} - Code","agent_type":"coder","priority":3},
        {"id":"4","description":f"{user_input} - Database","agent_type":"database","priority":4},
        {"id":"5","description":f"{user_input} - Test","agent_type":"tester","priority":5},
    ]
