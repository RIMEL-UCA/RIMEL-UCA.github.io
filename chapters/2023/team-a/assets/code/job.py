class Job:
    def __init__(self, name, dependencies, steps):
        self.name = name
        self.dependencies = dependencies
        self.steps = steps

    def __str__(self) -> str:
        return f'Job: {{name: {self.name}, dependencies: {self.dependencies}, step: {self.steps}}}'

    def __repr__(self) -> str:
        return f'Job: {{name: {self.name}, dependencies: {self.dependencies}, step: {self.steps}}}'

