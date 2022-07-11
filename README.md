# Thesis Recommender System

## Brief Description

This project was intended to explore the possibility of experimenting with changes to Thesis recommender system used in Masaryk University Information System (is.muni.cz). At the time the recommender system there relied only on keywords which was highly problematic in some cases (DNS is keyword in computer networks but also in physiotherapy). This project experimented with using additional metadata and text data (english abstract). Additionally, it was intended that everything is coded from scratch.

## Folder Structure

- `notebooks` - contains notebooks used for exploration
    - `analysis.ipynb` - initial data analysis
    - `model_demonstration.ipynb` - demonstration of models with projection visualisations, also some base statistical comparison
- `reports` - contains report on the project with some literature linked
- `LICENSE` - standard MIT license
- `requirements.txt` - requirements
- `bm25.py` - BM25+ implementation from scratch
- `encoders.py` - implementation of one hot encoding
- `load_data.py` - functionalities for storage and loading of data
- `preprocess.py` - preprocessing functions for text data
- `rssystem.py` - recommender system and scorer classes implemented (upper interface)
- `similarities.py` - similarity metrics implementation

## Outcomes

The outcomes of the project were that metadata features can be used to improve the keywords model. However, the use of text data (english abstract) did not lead to improvement! The suggestion was that maybe the use of features calculated from whole thesis text could lead to improvement.

## Notes

- The project was conducted in two people. The other person created evaluation web page and experimented with word embeddings. Only code the repository owner is available here (whole project is in Faculty Gitlab)
- The data is currently unavailable
