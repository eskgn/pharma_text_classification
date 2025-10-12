L’analyse pharmaceutique est une étape essentielle du circuit du médicament en milieu hospitalier. Lors de cette étape, les pharmaciens rédigent des interventions pharmaceutiques (IP) afin de signaler ou corriger des anomalies de prescription. Ces interventions sont ensuite classées selon la nomenclature de la Société Française de Pharmacie Clinique (SFPC), composée de 11 classes principales. Cependant, cette classification est manuelle, chronophage et sujette à variabilité inter-opérateur.

L’objectif de ce travail est de développer un modèle de traitement automatique du langage (NLP) capable de classifier automatiquement les interventions pharmaceutiques dans les 11 classes SFPC à partir du commentaire textuel rédigé par le pharmacien.

Un modèle de langue pré-entraîné francophone, CamemBERT, a été fine-tuné sur un corpus institutionnel constitué de métadonnées issues du logiciel d’aide à la prescription utilisé aux Hôpitaux Universitaires de Strasbourg. Les données comportaient trois champs : le libellé du médicament prescrit, le commentaire associé à l’intervention, et la classe SFPC correspondante. Une classification supervisée multi-classes a été effectuée à partir du seul champ textuel du commentaire.

Le modèle fine-tuné atteint un F1-score moyen de 0.82, démontrant une excellente capacité de généralisation malgré la complexité linguistique et la diversité des formulations utilisées par les pharmaciens.

Ces résultats montrent que l’utilisation de modèles de langue pré-entraînés permet d’automatiser efficacement la classification des interventions pharmaceutiques, ouvrant la voie à une standardisation et une accélération de la documentation clinique au sein des services de pharmacie hospitalière.
