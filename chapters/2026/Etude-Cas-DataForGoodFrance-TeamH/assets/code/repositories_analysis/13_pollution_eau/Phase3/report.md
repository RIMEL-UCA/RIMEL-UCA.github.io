# Phase 3

## Traitement des données

1. Insertion
    - Fichiers
        - `pipelines/tasks/client/core/duckdb_client.py`
    - Rôle : Insère dans DuckDB.

2. Transformation
    1. **Staging**
        - Fichiers
            - `dbt_/models/staging/edc/`
            - `dbt_/models/staging/communes/`
            - `dbt_/models/staging/atlasante/`
        - Opérations : Nettoyage basique, typage, renommage colonnes.
        - Rôle : Documente et valide données brutes.

    2. **Intermediate**
        - Fichiers
            - `dbt_/models/intermediate/int__resultats_udi_communes.sql`
            - `dbt_/models/intermediate/int__valeurs_de_reference.sql`
            - `dbt_/models/intermediate/int__udi.sql`
        - Opérations : Jointures complexes, agrégations par UDI/commune, calculs de conformité.
        - Rôle : Logique métier centrale.

    3. **Website**
        - Fichiers
            - `dbt_/models/website/web__resultats_udi.sql`
            - `dbt_/models/website/web__resultats_communes.sql`
            - `dbt_/models/website/web__stats_udi.sql`
        - Opérations : Pivot par période/catégorie, agrégations finales pour cartographie et stats.
        - Rôle : Modèles optimisés pour consommation front/PMTiles.

