import urllib.request
import urllib.parse
import urllib.error
import json
import csv
import sys
import base64
import time

# Récupération des arguments passés par le script bash
if len(sys.argv) < 4:
    print("Usage: python export_to_csv.py <PROJECT_KEY> <SONAR_TOKEN> <OUTPUT_FILE>")
    sys.exit(1)

PROJECT_KEY = sys.argv[1]
TOKEN = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
SONAR_URL = "http://sonarqube-server:9000"

auth_str = f"{TOKEN}:"
b64_auth = base64.b64encode(auth_str.encode()).decode()
headers = {"Authorization": f"Basic {b64_auth}"}

print(f"Utilisation du token pour le projet '{PROJECT_KEY}'")

def count_files_via_component_tree(project_key: str, page_size: int = 500) -> int:
    page = 1
    total_count = 0
    while True:
        url = (
            f"{SONAR_URL}/api/measures/component_tree"
            f"?component={urllib.parse.quote(project_key)}"
            f"&qualifiers=FIL"
            f"&ps={page_size}&p={page}"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

        components = data.get("components", []) or []
        total_count += len(components)

        paging = data.get("paging", {}) or {}
        total_items = int(paging.get("total", 0) or 0)
        page_index = int(paging.get("pageIndex", page) or page)
        page_size_resp = int(paging.get("pageSize", page_size) or page_size)

        if page_index * page_size_resp >= total_items or not components:
            break
        page += 1

    return total_count


def rating_to_score(v) -> int:
    # Convertit rating A..E ou 1..5 en score 0..100
    # Sonar renvoie souvent "1.0".."5.0" pour *_rating (A..E)
    # 1->A, 2->B, 3->C, 4->D, 5->E
    mapping_num = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    mapping_letter = {"A": 100, "B": 80, "C": 60, "D": 40, "E": 20}

    if v is None:
        return 0
    s = str(v).strip().upper()
    if s in mapping_letter:
        return mapping_letter[s]
    try:
        n = int(float(s))
        return mapping_num.get(n, 0)
    except Exception:
        return 0


def clamp_0_100(x: float) -> float:
    return max(0.0, min(100.0, x))


def sonar_get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def wait_for_measures(project_key: str, metric_keys: str, timeout_s: int = 180, step_s: int = 5) -> dict:
    """
    Attend que SonarQube retourne au moins une mesure (Compute Engine terminé).
    """
    url = f"{SONAR_URL}/api/measures/component?component={urllib.parse.quote(project_key)}&metricKeys={metric_keys}"
    deadline = time.time() + timeout_s

    last = None
    while time.time() < deadline:
        last = sonar_get_json(url)
        measures = last.get("component", {}).get("measures", []) or []
        if len(measures) > 0:
            return last
        print(f"   ... mesures pas prêtes (0 measure). Attente {step_s}s")
        time.sleep(step_s)

    raise RuntimeError(f"Timeout: aucune mesure dispo après {timeout_s}s pour {project_key}. Dernière réponse={last}")

def pick_measure(measures: dict, *keys):
    for k in keys:
        v = measures.get(k)
        if v is not None:
            return v
    return None

def fetch_project_score(project_key: str) -> dict:
    print(f"🔍 Récupération des métriques pour le projet '{project_key}'")

    metric_keys = (
        "software_quality_reliability_rating,"
        "software_quality_maintainability_rating,"
        "software_quality_security_rating,"
        "duplicated_lines_density,"
        "cognitive_complexity,"
        "files"
    )

    print(f"   → Récupération des mesures depuis {SONAR_URL} (attente si besoin)")
    data = wait_for_measures(project_key, metric_keys, timeout_s=240, step_s=5)

    measures = {m["metric"]: m.get("value") for m in data.get("component", {}).get("measures", [])}

    reliability_v = pick_measure(measures, "software_quality_reliability_rating")
    maintainability_v = pick_measure(measures, "software_quality_maintainability_rating")
    security_v = pick_measure(measures, "software_quality_security_rating")

    reliability = rating_to_score(reliability_v)
    maintainability = rating_to_score(maintainability_v)
    security = rating_to_score(security_v)
    try:
        duplication_density = float(measures.get("duplicated_lines_density") or 0.0)
    except Exception:
        duplication_density = 0.0
    duplication = max(0.0, 100.0 - duplication_density * 5)

    try:
        cognitive_global = float(measures.get("cognitive_complexity") or 0.0)
    except Exception:
        cognitive_global = 0.0

    # nb fichiers: via 'files' si présent, sinon fallback fiable
    try:
        nb_files = int(float(measures.get("files"))) if measures.get("files") is not None else 0
    except Exception:
        nb_files = 0

    if nb_files <= 0:
        # fallback sur component_tree si nécessaire
        nb_files = count_files_via_component_tree(project_key)

    avg_cognitive_per_file = (cognitive_global / nb_files) if nb_files > 0 else 0.0

    # score 0..100 à calibrer
    complexity_score = clamp_0_100(100.0 - (avg_cognitive_per_file * 5))

    score = (
        0.25 * reliability +
        0.20 * maintainability +
        0.15 * security +
        0.20 * duplication +
        0.20 * complexity_score
    )

    return {
        "project_key": project_key,
        "score": round(clamp_0_100(score), 2),
        "reliability": round(float(reliability), 2),
        "maintainability": round(float(maintainability), 2),
        "security": round(float(security), 2),
        "duplication": round(clamp_0_100(duplication), 2),
        "avg_cognitive_per_file": round(float(avg_cognitive_per_file), 2),
        "complexity": round(float(complexity_score), 2),
        "nb_files": int(nb_files),
        "cognitive_global": round(float(cognitive_global), 2),
    }

print(f"Récupération des scores pour {PROJECT_KEY}...")

try:
    score_info = fetch_project_score(PROJECT_KEY)
except Exception as e:
    print(f"Erreur: impossible de récupérer les métriques: {e}")
    sys.exit(2)

print(
    f"Score={score_info['score']:.2f}/100 "
    f"(R={score_info['reliability']:.0f}, M={score_info['maintainability']:.0f}, "
    f"S={score_info['security']:.0f}, D={score_info['duplication']:.2f})"
)

fieldnames = [
    "project_key",
    "score",
    "reliability",
    "maintainability",
    "security",
    "duplication",
    "complexity",
    "avg_cognitive_per_file",
    "nb_files",
    "cognitive_global",
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(score_info)

print(f"Export CSV: {OUTPUT_FILE}")