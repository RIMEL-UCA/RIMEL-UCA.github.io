from typing import TypedDict


class _Action(TypedDict):
    name: str
    parsers: list[str]


class _Project(TypedDict):
    git_url: str
    actions: list[_Action]


class RunConfig(TypedDict):
    projects: dict[str, _Project]
