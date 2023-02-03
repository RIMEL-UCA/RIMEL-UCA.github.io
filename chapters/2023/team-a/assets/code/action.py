from enum import Enum


class ActionType(Enum):
    GITHUB = 1,
    PERSONAL = 2,
    PUBLIC = 3


class Action:
    def __init__(self, name, action_type):
        self.name = name
        self.action_type = action_type

    def __str__(self) -> str:
        return f'Action: {{name: {self.name}, action_type: {self.action_type}}}'

    def __repr__(self) -> str:
        return f'Action: {{name: {self.name}, action_type: {self.action_type}}}'
