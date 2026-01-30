# Phase 2

## Types et sources des données

1. Données de qualité de l'eau EDC (Eau distribuée par commune)
    - Source : data.gouv.fr
    - Fichiers
        - `pipelines/tasks/client/datagouv_client.py`
        - `pipelines/tasks/config/config_edc.py`
    - Type : CSV/JSON structuré (résultats d'analyses par prélèvement : concentrations de pesticides, PFAS, nitrates, CVM, perchlorate).
    - Rôle : Base des indicateurs de qualité


2. Données géographiques des communes
    - Sources
        - OpenDataSoft
        - INSEE
    - Fichiers
        - `pipelines/tasks/client/opendatasoft_client.py`
        - `pipelines/tasks/config/config_geojson.py`
        - `pipelines/tasks/client/commune_client.py`
        - `pipelines/tasks/config/config_insee.py`
    - Type : JSON/GeoJSON
    - Rôle : Zones alternatives pour la cartographie


3. Données géographiques des unités de distribution (UDI) :
    - Source : Atlasanté (récupérer sur leur base de données s3)
    - Fichiers
        - `pipelines/tasks/client/uploaded_geojson_client.py`
        - `pipelines/tasks/config/config_uploaded_geojson.py`
    - Type : GeoJSON (géométries vectorielles : polygones des réseaux d'eau).
    - Rôle : Définit les zones pour la cartographie


4. Données de référence (valeurs de qualité) :
    - Source : Générations Futures (via seeds DBT : dbt_/seeds/).
    - Fichiers
        - `dbt_/seeds/references_generations_futures.csv`
        - `dbt_/seeds/udi_population_from_infofactures.csv`
    Type : CSV statique
    Rôle : Seuils pour classer les niveaux (conforme/non conforme).