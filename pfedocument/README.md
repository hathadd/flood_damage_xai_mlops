# PFE Document - Setup pour Overleaf

## Structure du dossier

```
pfedocument/
├── main.tex          # Document LaTeX principal (compilé sur Overleaf)
├── pdfs/             # Dossier pour vos PDFs (créer si nécessaire)
│   ├── evaluation_report.pdf
│   ├── model_architecture.pdf
│   └── ...
└── README.md         # Ce fichier
```

## Instructions d'utilisation

### 1. Créer le dossier `pdfs/`

Créez un sous-dossier `pdfs/` dans `pfedocument/` pour y stocker vos fichiers PDF.

```bash
mkdir -p pfedocument/pdfs
```

### 2. Ajouter vos PDFs

Placez vos fichiers PDF dans `pfedocument/pdfs/`.

Exemples :
- `evaluation_report.pdf`
- `model_architecture.pdf`
- `grad_cam_results.pdf`
- `confusion_matrices.pdf`

### 3. Intégrer les PDFs dans le document LaTeX

Modifiez `main.tex` dans la section `\appendix` pour ajouter vos PDFs :

```latex
\subsection{Rapport d'évaluation}
\includepdf[pages=-]{pdfs/evaluation_report.pdf}

\subsection{Architecture du modèle}
\includepdf[pages=-]{pdfs/model_architecture.pdf}

\subsection{Résultats Grad-CAM}
\includepdf[pages=-]{pdfs/grad_cam_results.pdf}
```

### 4. Options d'inclusion de PDFs

| Option | Effet |
|--------|-------|
| `\includepdf[pages=-]{pdfs/file.pdf}` | Inclure toutes les pages |
| `\includepdf[pages=1-5]{pdfs/file.pdf}` | Inclure pages 1 à 5 |
| `\includepdf[pages=3]{pdfs/file.pdf}` | Inclure uniquement la page 3 |
| `\includepdf[pages={1,3,5}]{pdfs/file.pdf}` | Inclure pages 1, 3, 5 uniquement |

### 5. Copier sur Overleaf

#### Option A : Upload direct

1. Allez sur https://www.overleaf.com/
2. Créez un nouveau projet
3. Cliquez sur "Upload Project"
4. Sélectionnez `/path/to/pfedocument/` ou zippez le dossier

#### Option B : Git Overleaf

1. Allez sur Overleaf → Settings
2. Copiez le Git repo link
3. Clonez localement :
   ```bash
   git clone https://git.overleaf.com/xxxxx
   cd xxxxx
   ```
4. Copiez le contenu de `pfedocument/` dans ce dossier
5. Commitez et pushez

#### Option C : Copier-coller le `main.tex`

1. Créez un projet Overleaf vide
2. Copiez le contenu de `main.tex` dans l'éditeur
3. Uploadez les PDFs via "Add files" → "Upload"

### 6. Compiler et exporter

Une fois sur Overleaf :

1. Cliquez sur "Recompile"
2. Attendez la génération du PDF
3. Téléchargez le PDF final via "Download PDF"

## Remarques importantes

- **Package `pdfpages`** : Assurez-vous que Overleaf a ce package (c'est le cas par défaut)
- **Chemins relatifs** : Utilisez toujours `pdfs/nom_du_fichier.pdf` (chemins relatifs)
- **Encodage** : Le fichier utilise UTF-8 ; cela fonctionne bien avec les caractères français
- **Taille des PDFs** : Overleaf supporte des fichiers volumineux, mais une accumulation peut ralentir la compilation

## Contenu actuel du document

Le `main.tex` fourni contient :
- Page de titre professionnelle
- Sommaire automatique
- Sections pré-remplies :
  - Introduction
  - Data Foundation
  - Pipeline d'apprentissage
  - Explicabilité (Grad-CAM)
  - MLOps et versionnement
  - Résultats et interprétation
  - Perspectives futures
  - Références
- Section vide pour les annexes PDF

## Adaptation du contenu

Libre de modifier les sections existantes pour correspondre à votre PFE :
- Remplacez les titres, noms d'auteur, dates
- Ajoutez/supprimez des sections selon vos besoins
- Intégrez vos résultats spécifiques
- Personnalisez les couleurs en changeant les définitions `\definecolor`

## Exemple complet d'ajout de PDFs

```latex
\appendix

\section{Annexes - Résultats détaillés}

\subsection{Rapport d'évaluation - Run B}
\includepdf[pages=-]{pdfs/run_b_evaluation.pdf}

\subsection{Matrices de confusion}
\includepdf[pages=1]{pdfs/confusion_matrices.pdf}

\subsection{Visualisations Grad-CAM}
\includepdf[pages=-]{pdfs/grad_cam_visualizations.pdf}

\subsection{Architecture détaillée}
\includepdf[pages=1-3]{pdfs/architecture_details.pdf}
```

## Support

Fichier LaTeX prêt à la production, compatible avec :
- Overleaf (recommandé)
- LaTeX local (MiKTeX, TeX Live)
- Tout éditeur LaTeX standard

Bonne rédaction de votre PFE ! 📚

