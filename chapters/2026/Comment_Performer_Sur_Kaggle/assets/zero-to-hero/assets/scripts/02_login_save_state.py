import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="kaggle_state.json",
        help="Chemin du fichier storage_state JSON à créer (par défaut: kaggle_state.json)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Lancer le navigateur en mode headless (par défaut: fenêtré)",
    )
    args = parser.parse_args()

    out_path = Path(args.out)

    print("[LOG] Lancement de Playwright pour récupérer une session Kaggle...")
    print(f"[LOG] Le storage_state sera sauvegardé dans : {out_path.resolve()}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()

        # Ouvre la page de login Kaggle
        print("[LOG] Ouverture de la page de connexion Kaggle...")
        page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded")

        print(
            "\n[ACTION] Connecte-toi à Kaggle dans la fenêtre qui vient de s’ouvrir "
            "(email/mot de passe, éventuellement Google, etc.)."
        )
        print(
            "[ACTION] Une fois que tu es connecté et que Kaggle te voit comme loggé, "
            "appuie sur Entrée dans ce terminal pour continuer.\n"
        )
        input(">>> Appuie sur Entrée ici quand tu as fini la connexion... ")

        # Optionnel : vérifier qu'on est bien connecté en allant sur la page d'accueil
        page.goto("https://www.kaggle.com", wait_until="domcontentloaded")

        # Sauvegarde du storage_state
        context.storage_state(path=str(out_path))
        print(f"[OK] Session sauvegardée dans : {out_path.resolve()}")

        browser.close()

    print("[LOG] Terminé. Tu peux maintenant utiliser ce fichier avec les autres scripts Playwright.")


if __name__ == "__main__":
    main()
