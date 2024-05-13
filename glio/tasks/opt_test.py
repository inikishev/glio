"""Tesa"""

class Task:
    def __init__(self, description, time_taken):
        self.description = description
        self.time_taken = time_taken
        self.completed = False

    def mark_completed(self):
        self.completed = True

    def get_time_taken(self):
        return self.time_taken
class Test:
    def __init__(self, task, optimizer):
        self.task = task
        self.optimizer = optimizer