class Step:
    def __init__(self, action):
        self.action = action

    def __str__(self) -> str:
        return f'Step: {{action: {self.action.__str__()}}}'

    def __repr__(self) -> str:
        return f'Step: {{action: {self.action.__str__()}}}'

