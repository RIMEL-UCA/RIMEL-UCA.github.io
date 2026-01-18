import pandas as pd

print("📊 Étape 2 : Création des fichiers par question de recherche")
print("=" * 80)

# Chargement des données de l'étape 1
df_etape1 = pd.read_csv('etape1_icc_par_projet.csv')

# Chargement du mapping
df_mapping = pd.read_csv('mapping_projets.csv')

# Fusion des données
df_merged = pd.merge(df_mapping, df_etape1, on='Projet', how='inner')

print(f"✅ {len(df_merged)} projets avec mapping chargés")
print(f"⚠️  Projets exclus (pas de mapping): {set(df_etape1['Projet']) - set(df_mapping['Projet'])}")

# ====================================================================
# QUESTION 1 : Impact de la PÉRIODE (Avant/Pendant/Après GenAI)
# ====================================================================
print("\n📝 Question 1 : Impact de la période temporelle")

q1_data = df_merged[['Période', 'Projet', 'Note moyenne test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'Interprétation ICC documentation']].copy()

# Tri par période
period_order = {'Avant GenAI': 0, 'Pendant GenAI': 1, 'Après GenAI': 2}
q1_data['_order'] = q1_data['Période'].map(period_order)
q1_data = q1_data.sort_values('_order').drop('_order', axis=1)

q1_data.to_csv('question1_impact_periode.csv', index=False, encoding='utf-8')
print(f"   ✅ question1_impact_periode.csv créé ({len(q1_data)} projets)")
print(f"   - Avant GenAI: {len(q1_data[q1_data['Période'] == 'Avant GenAI'])} projets")
print(f"   - Pendant GenAI: {len(q1_data[q1_data['Période'] == 'Pendant GenAI'])} projets")
print(f"   - Après GenAI: {len(q1_data[q1_data['Période'] == 'Après GenAI'])} projets")

# ====================================================================
# QUESTION 2 : Impact du VOLUME DE CONTRIBUTEURS (Peu/Beaucoup)
# ====================================================================
print("\n📝 Question 2 : Impact du volume de contributeurs")

q2_data = df_merged[['Volume_contributeurs', 'Projet', 'Note moyenne test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'Interprétation ICC documentation']].copy()

# Tri : Peu avant Beaucoup
volume_order = {'Peu': 0, 'Beaucoup': 1}
q2_data['_order'] = q2_data['Volume_contributeurs'].map(volume_order)
q2_data = q2_data.sort_values('_order').drop('_order', axis=1)

q2_data.to_csv('question2_impact_volume_contributeurs.csv', index=False, encoding='utf-8')
print(f"   ✅ question2_impact_volume_contributeurs.csv créé ({len(q2_data)} projets)")
print(f"   - Peu de contributeurs: {len(q2_data[q2_data['Volume_contributeurs'] == 'Peu'])} projets")
print(f"   - Beaucoup de contributeurs: {len(q2_data[q2_data['Volume_contributeurs'] == 'Beaucoup'])} projets")

# ====================================================================
# QUESTION 3 : Impact du TYPE (AI-related / Non AI-related)
# ====================================================================
print("\n📝 Question 3 : Impact du type de projet (AI vs non-AI)")

q3_data = df_merged[['Type_AI', 'Projet', 'Note moyenne test', 'Interprétation ICC test',
                      'Note moyenne documentation', 'Interprétation ICC documentation']].copy()

# Tri : AI-related avant Non AI-related
type_order = {'AI-related': 0, 'Non AI-related': 1}
q3_data['_order'] = q3_data['Type_AI'].map(type_order)
q3_data = q3_data.sort_values('_order').drop('_order', axis=1)

q3_data.to_csv('question3_impact_type_ai.csv', index=False, encoding='utf-8')
print(f"   ✅ question3_impact_type_ai.csv créé ({len(q3_data)} projets)")
print(f"   - AI-related: {len(q3_data[q3_data['Type_AI'] == 'AI-related'])} projets")
print(f"   - Non AI-related: {len(q3_data[q3_data['Type_AI'] == 'Non AI-related'])} projets")

# ====================================================================
# Statistiques descriptives par question
# ====================================================================
print("\n" + "=" * 80)
print("📊 STATISTIQUES DESCRIPTIVES")
print("=" * 80)

print("\n🔬 Q1 - Par période:")
stats_q1 = df_merged.groupby('Période').agg({
    'Note moyenne test': ['mean', 'std', 'count'],
    'Note moyenne documentation': ['mean', 'std', 'count']
}).round(2)
print(stats_q1)

print("\n🔬 Q2 - Par volume de contributeurs:")
stats_q2 = df_merged.groupby('Volume_contributeurs').agg({
    'Note moyenne test': ['mean', 'std', 'count'],
    'Note moyenne documentation': ['mean', 'std', 'count']
}).round(2)
print(stats_q2)

print("\n🔬 Q3 - Par type de projet:")
stats_q3 = df_merged.groupby('Type_AI').agg({
    'Note moyenne test': ['mean', 'std', 'count'],
    'Note moyenne documentation': ['mean', 'std', 'count']
}).round(2)
print(stats_q3)

print("\n" + "=" * 80)
print("✨ Étape 2 terminée avec succès !")
print("=" * 80)
