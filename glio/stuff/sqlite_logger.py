import sqlite3, math

class DictLogger:
    def __init__(self):
        self.logs = {}

    def log(self, step, metric, value):
        if metric not in self.logs: self.logs[metric] = {step: value}
        else: self.logs[metric][step] = value

class SQLiteLogger:
    def __init__(self, file):
        """10 times slower than dict logger!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
        self.con = sqlite3.connect(file)
        self.cur = self.con.cursor()
        self.cur.execute("CREATE TABLE logs (step INTEGER PRIMARY KEY)")
        self.current_step = float('nan')
        self.metrics_to_add = {}
        self.columns = []

    def log(self, step, metric, value):
        if math.isnan(self.current_step): self.current_step = step
        if step == self.current_step:
            self.metrics_to_add[metric] = value
            if metric not in self.columns:
                self.columns.append(metric)
                self.cur.execute(f"ALTER TABLE logs ADD COLUMN {metric} FLOAT")
        else:
            self.cur.execute(
                f"INSERT INTO logs (step, {', '.join(self.metrics_to_add.keys())}) VALUES ({step}, {', '.join([str(i) for i in self.metrics_to_add.values()])})",)
            self.current_step = step
            self.metrics_to_add = {metric: value}
            if metric not in self.columns:
                self.columns.append(metric)
                self.cur.execute(f"ALTER TABLE logs ADD COLUMN {metric} FLOAT")

