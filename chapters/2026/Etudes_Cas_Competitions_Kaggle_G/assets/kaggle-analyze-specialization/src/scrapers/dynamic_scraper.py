"""Scraper pour les sites dynamiques utilisant Playwright."""

import asyncio
from typing import Any

from playwright.async_api import async_playwright, Page, Browser
from bs4 import BeautifulSoup

from .base import BaseScraper


class DynamicScraper(BaseScraper):
    """Scraper pour les pages avec contenu JavaScript dynamique."""

    def __init__(
        self,
        base_url: str,
        headers: dict | None = None,
        headless: bool = True,
        timeout: int = 30000,
        wait_for_selector: str | None = None,
    ):
        super().__init__(base_url, headers)
        self.headless = headless
        self.timeout = timeout
        self.wait_for_selector = wait_for_selector
        self._browser: Browser | None = None
        self._page: Page | None = None

    async def _init_browser(self) -> None:
        """Initialise le navigateur Playwright."""
        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch(headless=self.headless)
        context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            extra_http_headers=self.headers,
        )
        self._page = await context.new_page()

    async def _close_browser(self) -> None:
        """Ferme le navigateur."""
        if self._browser:
            await self._browser.close()

    async def fetch_async(self, url: str) -> str:
        """Recupere le contenu HTML d'une URL de maniere asynchrone."""
        if not self._page:
            await self._init_browser()

        await self._page.goto(url, timeout=self.timeout)

        if self.wait_for_selector:
            await self._page.wait_for_selector(
                self.wait_for_selector, timeout=self.timeout
            )

        return await self._page.content()

    def fetch(self, url: str) -> str:
        """Recupere le contenu HTML (wrapper synchrone)."""
        return asyncio.get_event_loop().run_until_complete(self.fetch_async(url))

    def parse(self, html: str) -> list[dict[str, Any]]:
        """
        Parse le HTML. A surcharger dans les classes filles.

        Exemple d'implementation:
            soup = self.get_soup(html)
            items = soup.select(".dynamic-content")
            return [{"data": item.text} for item in items]
        """
        return []

    def get_soup(self, html: str) -> BeautifulSoup:
        """Retourne un objet BeautifulSoup pour parser le HTML."""
        return BeautifulSoup(html, "lxml")

    async def scrape_async(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape une liste d'URLs de maniere asynchrone."""
        try:
            await self._init_browser()
            for url in urls:
                html = await self.fetch_async(url)
                extracted = self.parse(html)
                self.data.extend(extracted)
            return self.data
        finally:
            await self._close_browser()

    def scrape(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape une liste d'URLs (wrapper synchrone)."""
        return asyncio.get_event_loop().run_until_complete(self.scrape_async(urls))

    async def screenshot(self, url: str, filepath: str) -> None:
        """Prend une capture d'ecran d'une page."""
        if not self._page:
            await self._init_browser()
        await self._page.goto(url, timeout=self.timeout)
        await self._page.screenshot(path=filepath, full_page=True)

    async def scroll_to_bottom(self, pause: float = 1.0) -> None:
        """Scroll jusqu'en bas de la page (utile pour le lazy loading)."""
        if not self._page:
            return

        prev_height = 0
        while True:
            current_height = await self._page.evaluate("document.body.scrollHeight")
            if current_height == prev_height:
                break
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(pause)
            prev_height = current_height
