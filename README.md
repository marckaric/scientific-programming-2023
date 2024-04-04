# Analyse und Clustering des Slurm Datensatzes

Dieses Projekt besteht aus fünf Jupyter-Notebooks und enthält eine Datenbereinigung und -exploration, eine Skalierung und Dimensionsreduktion der Daten und eine Clusteranalyse. Es folgt eine Untersuchung der Cluster auf wesentliche Unterschiede und ein Training zweier Klassifizierer auf die Cluster.
Ordnerstruktur

### Das Projekt ist in folgende Jupyter-Notebooks unterteilt:

    data_setup.ipynb: Grundlegende Datenbereinigung und Übersicht und Korrektur der Datentypen
    data_exploration.ipynb: Einfache EDA der Daten und Erwähnung von Auffälligkeiten
    data_pipeline.ipynb: Skalierung und Dimensionsreduktion der Daten und Clustereinteilung
    cluster_exploration.ipynb: Untersuchung der Cluster auf wesentliche Unterschiede unter den Clustern
    classification.ipynb: Training zweier Klassifizierer auf die Cluster und Vergleich

### Zusätzlich enthält das Projekt folgende Ordner und Dateien:

    figures: Diverse exportierte Abbildungen und Plots aus den Jupyter-Notebooks
    models: Visualisierung der Entscheidungsbäume aus dem classification-Notebook
    task_and_data: Aufgabenstellung und Daten
    tweaked_datasets: Mehrere csv-Dateien, die durch die Jupyter Notebooks entstehen
    presentation: Notebooks, figures und html-Dateien für die Präsentation
    help_funcs.py: Python-Datei, die einige Helferfunktionen enthält, die in den Notebooks verwendet werden.

## Verwendung

Um das Projekt auszuführen, empfiehlt es sich, die Notebooks in der oben angegebenen Reihenfolge auszuführen. Jedes Notebook erzeugt ein Ergebnis, das von den nachfolgenden Notebooks verwendet wird.

Die Daten können im Ordner "task_and_data" gefunden werden. Alle generierten Ergebnisse werden in separaten Ordnern gespeichert, um die Nachvollziehbarkeit des Projekts zu erleichtern. Darüber hinaus enthält das Projekt auch Abbildungen und Plots, die im Ordner "figures" gespeichert sind.

Zusätzlich wurden Helferfunktionen in die Datei "help_funcs.py" ausgelagert, um die Notebooks übersichtlicher zu gestalten.

## Autor

Dieses Projekt wurde von Marc Karic und Daniel Lückhof im Rahmen des Moduls "Wissenschaftliche Programmierung und Datenanalyse" durchgeführt.
Die Arbeit wurde folgendermaßen aufgeteilt:
    data_setup: gemeinsame Arbeit
    data_exploration: Marc Karic
    data_pipeline: Daniel Lückhof
    cluster_exploration: Daniel Lückhof
    classification: Marc Karic
