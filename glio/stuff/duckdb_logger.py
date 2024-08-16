import math
import duckdb
class DuckDBLogger:
    def __init__(self):
        """Ultra Slow"""
        self.con = duckdb.connect()
        self.con.sql("CREATE TABLE logs (step INTEGER PRIMARY KEY)")
        self.current_step = float('nan')
        self.metrics_to_add = {}
        self.columns = []

    def log(self, step, metric, value):
        if math.isnan(self.current_step): self.current_step = step
        if step == self.current_step:
            self.metrics_to_add[metric] = value
            if metric not in self.columns: 
                self.columns.append(metric)
                self.con.sql(f"ALTER TABLE logs ADD COLUMN {metric} FLOAT")
        else:
            self.con.sql(
                f"INSERT INTO logs (step, {', '.join(self.metrics_to_add.keys())}) VALUES ({step}, {', '.join([str(i) for i in self.metrics_to_add.values()])})",)
            self.current_step = step
            self.metrics_to_add = {metric: value}
            if metric not in self.columns: 
                self.columns.append(metric)
                self.con.sql(f"ALTER TABLE logs ADD COLUMN {metric} FLOAT")

