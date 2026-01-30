import asyncio
from typing import Any

from src.config import DATA_DIR
from src.scrapers import DynamicScraper


class KaggleScrapper(DynamicScraper):

    def __init__(self):
        super().__init__(
            base_url="https://www.kaggle.com",
            wait_for_selector="table.MuiTable-root",
            headless=True
        )

    def parse(self, html: str) -> list[dict[str, Any]]:
        """Parse le leaderboard et extrait les infos de chaque joueur."""
        tops = []
        soup = self.get_soup(html)

        for top_row_tr in soup.select(".MuiTableRow-root"):
            top_name_cell = top_row_tr.select_one("td:nth-child(3)")
            top_date_cell = top_row_tr.select_one("td:nth-child(4)")
            top_medals_cell = top_row_tr.select_one("td:nth-child(5)")
            top_score_cell = top_row_tr.select_one("td:last-child")



            if not top_name_cell:
                continue

            number_medals = 0
            if top_medals_cell:
                for medal in top_medals_cell.select("span"):
                    try:
                        number_medals += int(medal.get_text(strip=True))
                    except ValueError:
                        pass

            tops.append({
                "username": top_name_cell.select_one('a').attrs.get('href').replace("/", ""),
                "date": top_date_cell.get_text(strip=True) if top_date_cell else None,
                "medals": number_medals if number_medals else None,
                "score": top_score_cell.get_text(strip=True) if top_score_cell else None,
            })

        return tops

    async def fetch_profile_models(self, username: str) -> dict[str, Any]:
        profile_url = f"{self.base_url}/{username}/models"

        old_selector = self.wait_for_selector
        old_to = self.timeout

        self.wait_for_selector = "[data-testid=profile-render-tid]"
        self.timeout = 2500

        try:
            html = await self.fetch_async(profile_url)
            soup = self.get_soup(html)

            user_meta = soup.select(".sc-fbQrwq.sc-iUCBFZ.kHxwkV.hEemYU")

            if len(user_meta) == 2:
                location = user_meta[0].get_text(strip=True)
            else:
                location = user_meta[1].get_text(strip=True)

            return {
                "location": location
            }
        except Exception as e:
            print(f"Erreur lors du scraping du profil {username}: {e}")
            return {}
        finally:
            self.wait_for_selector = old_selector
            self.timeout = old_to

    async def scrape_with_profiles(
        self,
        leaderboard_url: str,
        fetch_profiles: bool = True,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        try:
            await self._init_browser()

            print(f"Scraping du leaderboard: {leaderboard_url}")
            html = await self.fetch_async(leaderboard_url)
            players = self.parse(html)
            print(f"Nombre de joueurs trouvés: {len(players)}")

            if fetch_profiles:
                players_to_fetch = players[:limit] if limit else players

                for i, player in enumerate(players_to_fetch):
                    username = player.get("username")
                    if username:
                        print(f"Scraping profil {i+1}/{len(players_to_fetch)}: {username}")
                        profile_data = await self.fetch_profile_models(username)
                        player.update(profile_data)

            self.data = players
            return players

        finally:
            await self._close_browser()

    def run_with_profiles(
        self,
        leaderboard_url: str,
        fetch_profiles: bool = True,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        return asyncio.run(
            self.scrape_with_profiles(leaderboard_url, fetch_profiles, limit)
        )


def main():
    scraper = KaggleScrapper()

    data = scraper.run_with_profiles(
        "https://www.kaggle.com/rankings",
        fetch_profiles=True,
        limit=100
    )

    print(data)

    scraper.to_csv(DATA_DIR / "tops.csv")
    scraper.to_json(DATA_DIR / "tops.json")

if __name__ == "__main__":
    main()
