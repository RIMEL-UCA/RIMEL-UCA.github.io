import pandas as pd

class ContributorClassifier:
    """
    Assigns a role based on contribution proportions using weighted scoring.
    """
    def __init__(self):
        self.weights = {
            "Frontend Engineer": {"frontend": 1.0},
            "Backend Engineer": {"backend": 1.0},
            "Fullstack Engineer": {"frontend": 0.7, "backend": 0.7},
            "Data Engineer": {"data": 0.7, "db": 0.7, "backend": 0.3, "infra": 0.3},
            "Data Scientist": {"exploration": 0.7, "ml": 0.5, "data": 0.2},
            "MLOps Engineer": {"ml": 0.5, "infra": 0.5},
            "DevOps / SRE": {"infra": 0.8, "config": 0.2},
            "QA / Test Engineer": {"testing": 1.0},
            "Technical Writer": {"docs": 1.0},
            "Maintainer": {"infra": 0.5, "config": 0.3, "docs": 0.3}
        }

    def infer(self, row: pd.Series) -> str:
        total = row.sum()
        if total == 0:
            return "Inactive"

        stats = row / total
        scores = {}
        for role, w in self.weights.items():
            score = sum(stats.get(col, 0) * weight for col, weight in w.items())
            scores[role] = score

        best_role = max(scores, key=scores.get)
        if scores[best_role] < 0.25:
            return "Generalist"
        return best_role