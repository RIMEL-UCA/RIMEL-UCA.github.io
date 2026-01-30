"""Classe de base pour les scrapers."""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class BaseScraper(ABC):
    """Classe abstraite definissant l'interface commune des scrapers."""

    def __init__(self, base_url: str, headers: dict | None = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.data: list[dict[str, Any]] = []

    @abstractmethod
    def fetch(self, url: str) -> str:
        """Recupere le contenu HTML d'une URL."""
        pass

    @abstractmethod
    def parse(self, html: str) -> list[dict[str, Any]]:
        """Parse le HTML et extrait les donnees."""
        pass

    def scrape(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape une liste d'URLs."""
        for url in urls:
            html = self.fetch(url)
            extracted = self.parse(html)
            self.data.extend(extracted)
        return self.data

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit les donnees en DataFrame pandas."""
        return pd.DataFrame(self.data)

    def to_csv(self, filepath: str) -> None:
        """Exporte les donnees en CSV."""
        self.to_dataframe().to_csv(filepath, index=False)

    def to_excel(self, filepath: str) -> None:
        """Exporte les donnees en Excel."""
        self.to_dataframe().to_excel(filepath, index=False)

    def to_json(self, filepath: str) -> None:
        """Exporte les donnees en JSON."""
        self.to_dataframe().to_json(filepath, orient="records", indent=2)
