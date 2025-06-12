📚 LightFM Buch-Empfehlungssystem
Ein hybrides Empfehlungssystem mit der LightFM-Bibliothek zur Empfehlung von Büchern auf Basis von Benutzer-Item-Interaktionen und eigenschaftsbasierten Metadaten.

# Projektübersicht
Dieses Projekt zielt darauf ab, ein Empfehlungssystem zu entwickeln, das Benutzerpräferenzen für Bücher auf Grundlage von impliziten Bewertungen vorhersagt. LightFM wird verwendet, da es kollaboratives Filtern mit inhaltsbasierten Features kombiniert, was besonders bei spärlichen Daten oder Kaltstartproblemen hilfreich ist.

# Vorgehensweise
## Feature Engineering für Bücher
Jedes Buch wurde in einen reichhaltigen Feature-Vektor umgewandelt, der für das hybride LightFM-Modell verwendet wird. Schritte:
1. Umgang mit fehlenden Werten:
Fehlende Autoren → "Unknown"
Fehlendes Erscheinungsjahr → mit Median aufgefüllt
Fehlender Sprachcode → "eng" als Standardwert
Fehlende durchschnittliche Bewertung → mit Mittelwert ersetzt

2. Kategorische Einteilung:
Bewertung → in niedrig, mittel, hoch, exzellent eingeteilt
Erscheinungsjahr → von „klassisch“ bis „zeitgenössisch“ gruppiert
Popularität (basierend auf Bewertungsanzahl) → „nischig“, „moderat“, „populär“, „Bestseller“

3. Erstellte Features:
Format: author:J_K_Rowling, language:eng, rating:high_rating usw.
Diese Features wurden als Item-Metadaten für LightFM verwendet

## Interaktionsdaten (Hybrides Feedback)
Benutzer-Buch-Interaktionen wurden aus Bewertungen abgeleitet und mit Vertrauensgewichten ergänzt:
1. Implizites Feedback:
Bewertungen ≥ 4 → positive Interaktion (1)
Bewertungen < 4 → ignoriert (0)

2. Vertrauensgewichtung (Confidence):
Normalisierte Bewertungswerte (z. B. Bewertung 4 = 0.8, Bewertung 5 = 1.0)
Diese Werte wurden als Gewichtung für die Interaktionen an LightFM übergeben

🔹 Datensatz-Erstellung
1. Erstellt mit:
Eindeutigen Benutzer- und Buch-IDs
Benutzerdefinierten Buch-Features aus Metadaten

2. Gebaute Matrizen:
Interaktionsmatrix mit positivem impliziten Feedback und gewichteten Bewertungen
Item-Feature-Matrix für hybride Empfehlungen

# Modellvergleich
Drei LightFM-Loss-Funktionen wurden getestet:

Modell	Precision	Recall	NDCG	AUC
WARP	0.1248	0.0059	0.129	0.9529
BPR	0.0905	0.0043	0.113	0.9022
WARP-KOS	0.1190	0.0056	0.108	0.9115

✅ WARP zeigte die beste Leistung, insbesondere bei NDCG, Precision und AUC, und wurde daher für das finale Training gewählt.

# Hyperparameter-Tuning
Durch Randomized Search wurden folgende optimale Parameter gefunden:
{
  'loss': 'warp',
  'no_components': 24,
  'learning_rate': 0.0959,
  'item_alpha': 4.37e-05
}


# Finales Modell
Das finale LightFM-Modell wurde wie folgt trainiert:

model = LightFM(
    loss='warp',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    no_components=50,
    random_state=42
)


# Evaluationsergebnisse
Precision@10 und AUC:
Datensatz	Precision@10	AUC
Training	0.4426	0.9699
Test	0.1825	0.9565

Top-K-Evaluation (k=1 bis 20):
k	Precision@k	Recall@k	NDCG@k	AUC
1	0.2807	0.0191	0.1134	0.9565
5	0.2168	0.0725	0.1092	0.9565
10	0.1825	0.1207	0.1069	0.9565
20	0.1468	0.1922	0.1188	0.9565

📊 Diese Ergebnisse zeigen, dass das Modell besonders gut bei Ranking-basierten Empfehlungen (Top-N) abschneidet.

📊 Bedeutung der Scores
Das System gibt zwei verschiedene Arten von Scores aus – je nach Anwendungsfall:

# Scores
## Prediction Score (Benutzerbasierte Empfehlung)
Berechnet mit: model.predict()
Zweck: Schätzt, wie sehr ein Benutzer ein Buch mögen wird
Wertebereich: Unbegrenzt (höher = höhere Vorhersagepräferenz)
Verwendung:
recommend_books_for_user(user_id=1)

## Similarity Score (Item-basierte Ähnlichkeit)
Berechnet mit: cosine_similarity() auf Item-Embeddings
Zweck: Misst die Ähnlichkeit zwischen zwei Büchern im latenten Raum
Wertebereich: -1 bis 1 (1 = identisch, 0 = unabhängig)
Verwendung:
recommend_similar_books(book_id=11)


# Fazit
WARP war die leistungsstärkste Verlustfunktion für dieses implizite Feedback-Datenset.

Durch umfangreiches Feature Engineering und Hyperparameter-Tuning wurde die Empfehlungsqualität stark verbessert.

Das finale Modell bietet eine gute Balance zwischen Genauigkeit und Ranking-Leistung.