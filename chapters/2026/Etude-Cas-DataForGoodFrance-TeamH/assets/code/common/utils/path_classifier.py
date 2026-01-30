import re
from typing import List, Tuple, Pattern, Dict
from .categories import INFRA, DB, DOCS, TESTING, EXPLORATION, ML, DATA, FRONTEND, BACKEND, CONFIG

class PathClassifier:
    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self) -> List[Tuple[str, Pattern]]:
        
        # Helper: Matches a directory name exactly (start/end of string or surrounded by slashes)
        # matches: "back", "/back/", "src/back/api"
        # rejects: "background", "backup"
        def d(keywords: str) -> str:
            return fr"(?:^|/)({keywords})(?:$|/)"

        # Helper: Matches file extensions
        def e(exts: str) -> str:
            return fr"(\.({exts}))$"

        # Helper: Matches specific filenames exactly
        def f(names: str) -> str:
            return fr"(?:^|/)({names})$"

        return [
            # 1. Infrastructure & DevOps (Highest Priority: CI files inside backend are still infra)
            (INFRA, re.compile(
                d(r"\.github|ci|cd|workflows|k8s|kubernetes|terraform|ansible|infra|ops|deploy|docker|vagrant|nginx|build") + "|" + 
                f(r"dockerfile|docker-compose[\w.-]*|makefile|jenkinsfile|vagrantfile"), 
                re.IGNORECASE
            )),


            # 2. Documentation
            (DOCS, re.compile(
                d(r"docs?|documents?|wiki|man|manuals?|guides?|assets|tutorials?|examples?") + "|" + 
                e(r"md|rst|txt|pdf|adoc") + "|" +
                f(r"readme[\w.]*|license[\w.]*|contributing[\w.]*|changelog[\w.]*"), 
                re.IGNORECASE
            )),

            # 3. Testing (Overrides generic code matches)
            (TESTING, re.compile(
                d(r"tests?|tst|specs?|__tests__|__mocks__|e2e|it|integration|unit") + "|" + 
                r"(_test|\.test|\.spec|_spec)", # Capture suffixes like user_test.py
                re.IGNORECASE
            )),

            # 4. Data Science & ML
            (EXPLORATION, re.compile(d(r"notebooks?|explorations?|analysis|scratches") + "|" + e(r"ipynb"), re.IGNORECASE)),
            (ML, re.compile(
                d(r"models?|ml|dl|ai|weights?|checkpoints?|torch|tf|tensorflow|keras|sklearn|huggingface") + "|" + 
                e(r"pt|pth|h5|onnx|pkl|tflite") + "|" +
                f(r".*(?:train|predict|inference|evaluate|hparams|hyperparameters).*\.py"),
                re.IGNORECASE
            )),
            (DATA, re.compile(
                d(r"data|datasets?|processed|raw|corpus|corpora|scraping") + "|" + 
                e(r"csv|parquet|avro|feather|xlsx") + "|" +
                f(r".*(?:etl|process|clean|ingest|scrap).*\.py"),
                re.IGNORECASE
            )),
            
            # 6. Database & Migrations
            (DB, re.compile(
                d(r"db|database|sql|alembic|migrations|seeds?|schemas?|queries") + "|" + 
                e(r"sql|db|sqlite|sqlite3"), 
                re.IGNORECASE
            )),

            # 7. Frontend (Assets/Code)
            (FRONTEND, re.compile(
                d(r"frontend|front|fe|client|ui|web|www|public|static|styles|css|scss|less|views|pages|screens|components|containers|atoms|molecules|hooks|utils/fe") + "|" + 
                e(r"jsx|tsx|css|scss|less|sass|html|htm|vue|svelte|ico|svg|png|jpg|jpeg|webp"), 
                re.IGNORECASE
            )),

            # 8. Backend (Source Code)
            (BACKEND, re.compile(
                d(r"backend|back|be|server|srv|api|services|svc|controllers|routes|routers|middlewares|models|entities|dto|repositories|core|lib|pkg|internal|cmd|utils|common|shared") + "|" + 
                e(r"py|go|java|rb|php|rs|cs|c|cpp|h|hpp|pl|swift|kt"), 
                re.IGNORECASE
            )),

            # 9. Configuration
            (CONFIG, re.compile(
                d(r"configs?|conf|cfg|settings|env|parameters") + "|" + 
                e(r"ya?ml|toml|ini|json|json5|xml|properties|env(\..+)?|lock|gitignore|editorconfig|eslint.*|prettier.*|babel.*|tsconfig.*"), 
                re.IGNORECASE
            )),
        ]

    def classify(self, path: str) -> str:
        clean_path = path.replace("\\", "/").lower()
        
        for category, pattern in self.rules:
            if pattern.search(clean_path):
                return category
        
        return "other"


if __name__ == "__main__":
    classifier = PathClassifier()
    
    dataset = [
        "back/src/main.py",                   # -> backend
        "be/api/user.go",                     # -> backend
        "fe/src/app.tsx",                     # -> frontend
        "conf/prod.yml",                      # -> config
        
        "packages/services/auth/back/index.ts", # -> backend (matches 'back' dir)
        "apps/client/src/components/List.vue",  # -> frontend (matches 'client' and .vue)
        
        "backend/tests/unit/user_test.py",    # -> testing (overrides backend)
        "frontend/Dockerfile",                # -> infra (overrides frontend)
        "backend/alembic/versions/123.py",    # -> db (overrides backend/py)
        
        "config/config_t.yml",                # -> config
        "back/backend/server.py",             # -> backend
        
        "background/image.png",               # -> frontend (png) or other. (NOT backend)
        "backup/data.sql",                    # -> db (sql). (NOT backend via 'back')
    ]

    print(f"{'Path':<50} | {'Category'}")
    print("-" * 65)
    for p in dataset:
        print(f"{p:<50} | {classifier.classify(p)}")