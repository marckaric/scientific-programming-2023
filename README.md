# Analyse und Clustering des Slurm Datensatzes

Dieses Projekt besteht aus fünf Jupyter-Notebooks und enthält eine Datenbereinigung und -exploration, eine Skalierung und Dimensionsreduktion der Daten und eine Clusteranalyse. Es folgt eine Untersuchung der Cluster auf wesentliche Unterschiede und ein Training zweier Klassifizierer auf die Cluster.
Ordnerstruktur

### Das Projekt ist in folgende Jupyter-Notebooks unterteilt:

    data_setup.ipynb: Grundlegende Datenbereinigung und Übersicht und Korrektur der Datentypen
    data_exploration.ipynb: Einfache EDA der Daten und Erwähnung von Auffälligkeiten
    data_pipeline.ipynb: Skalierung und Dimensionsreduktion der Daten und Clustereinteilung
    cluster_exploration.ipynb: Untersuchung der Cluster auf wesentliche Unterschiede unter den Clustern
    classification.ipynb: Training zweier Klassifizierer auf die Cluster und Vergleich

## Verwendung

Um das Projekt auszuführen, empfiehlt es sich, die Notebooks in der oben angegebenen Reihenfolge auszuführen. Jedes Notebook erzeugt ein Ergebnis, das von den nachfolgenden Notebooks verwendet wird.
Da der Datensatz zu groß ist, ist dieser jedoch nicht in diesem Repository hinterlegt.
Zusätzlich wurden Helferfunktionen in die Datei "help_funcs.py" ausgelagert, um die Notebooks übersichtlicher zu gestalten.

## Autor

Dieses Projekt wurde von Marc Karic und Daniel Lückhof im Rahmen des Moduls "Wissenschaftliche Programmierung und Datenanalyse" durchgeführt.
Die Arbeit wurde folgendermaßen aufgeteilt:
    data_setup: gemeinsame Arbeit
    data_exploration: Marc Karic
    data_pipeline: Daniel Lückhof
    cluster_exploration: Daniel Lückhof
    classification: Marc Karic
