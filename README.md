# Mentor-Mentee Matching Tool

An end-to-end, data-driven web application for automating mentor-mentee group formation based on hobbies, personality traits, and availability constraints. Built with Python and Streamlit, this tool clusters participants, balances group sizes, and assigns mentors—all while keeping data entirely local and secure.

## 📋 Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Data & Pipeline](#data--pipeline)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Directory Structure](#directory-structure)    

## 🚀 Features
- **Data Upload**: CSV upload for mentee & mentor profiles  
- **NLP Processing**: spaCy-based extraction of hobbies (noun/verb lemmatization)  
- **Clustering**  
  - K-Means for preliminary hobby clusters  
  - Spectral Clustering on combined affinity (traits + availability)  
- **Refinement**:  
  - Enforce min/max group sizes  
  - Balance demographics & availability overlap  
- **Mentor Assignment**: Assigns 1–2 mentors per group respecting their availability limits  
- **Interactive UI**:  
  - Filter & sort clusters by size, gender, program, availability  
  - Manual reassignment of any mentee between clusters  
  - Visual statistics (heatmaps, word clouds)  

## 🛠️ Tech Stack
- **Language**: Python 3.8+  
- **Web/UI**: [Streamlit](https://streamlit.io/)  
- **NLP**: [spaCy](https://spacy.io/) (`en_core_web_md`)  
- **Data Handling**: pandas, numpy  
- **Clustering & ML**: scikit-learn (KMeans, SpectralClustering, AgglomerativeClustering)  
- **Visualization**: matplotlib, seaborn, WordCloud  
- **Dev & Packaging**: venv, pip, batch/shell scripts  

## 🔄 Data & Pipeline
1. **Upload CSVs**  
2. **Extract & Preprocess**  
   - `Hobbies_original` → spaCy noun/verb tokens  
   - Personality traits + availability  
3. **Initial Hobby Clustering** (K-Means)  
4. **Affinity Matrix** = trait similarity + availability overlap  
5. **Spectral Clustering** on combined affinity  
6. **Cluster Refinement**  
   - Enforce min/max size  
   - Reassign mismatches by distance & availability  
7. **Mentor Matching**  
   - Respect each mentor’s max groups & time slots  
8. **Manual Adjustments** via Streamlit UI  

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/mentor-mentee-matching.git
cd mentor-mentee-matching
# Windows
.\run_app.bat
# macOS/Linux
./run_app.command
```
## ▶️ Usage

1. Launch the app via the script above.
2. Navigate to Data Input to upload your CSVs and set group parameters.
3. Click Match to generate clusters.
4. Go to Clustering and Matching to review, filter, sort, or manually adjust groups.
5. Explore Statistics and Hobbies tabs for visual summaries.

## 📂 Directory Structure

```bash
.
├── app.py                 # Streamlit front-end
├── matchAlgo.py           # Clustering & matching logic
├── requirements.txt
├── run_app.bat            # Windows launcher script
├── run_app.command        # macOS/Linux launcher script
└── README.md

```
