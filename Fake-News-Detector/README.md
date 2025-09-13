# Real vs. Fake News Detection

As misinformation becomes more pervasive online, distinguishing fake news from real news is critical.  
This project investigates whether **machine learning models**, leveraging **natural language processing (NLP)** and statistical features, can effectively classify news articles as real or fake.

## Description
- Uses the **WELFake** dataset of labeled real and fake news articles.
- Engineers features such as article length, headline punctuation, and sentiment.
- Applies **Term Frequency–Inverse Document Frequency (TF-IDF)** to represent text data numerically.
- Implements multiple classifiers (Logistic Regression, Naïve Bayes, and ensemble stacking) and evaluates them with **Monte Carlo cross-validation**.
- Achieves **96.1% accuracy** and an **AUC-ROC of 0.993** with the ensemble model, showing that combining stylistic and linguistic cues with strong modeling offers a viable method for automatic fake news detection.

## Files
- `Final_Real_Fake_News_Detection.ipynb` – Main notebook containing all code and analysis.
- `cleaned_df.csv` – Preprocessed dataset (ready to use).
- `WELFake_Dataset.csv` – Original dataset (requires preprocessing).

All data files can be downloaded from this Google Drive folder:  
[Google Drive Link](https://drive.google.com/drive/folders/1KO5u5UOStEIiM0SHBO2s5BI6f06AmZjk?usp=drive_link)

## How to Open and Run
1. **Download the data**  
   Download `cleaned_df.csv` and `WELFake_Dataset.csv` from the Google Drive link above.  
   Place them in the **same folder** as `Final_Real_Fake_News_Detection.ipynb`.
2. **Open the notebook**  
   Use your preferred IDE (e.g., **Jupyter Notebook**, **JupyterLab**, or **VS Code** with Jupyter support).

3. **Choose one of the following approaches**  
   - **Option A – Run with the cleaned dataset** (faster):  
     Skip code blocks 2–4 and directly run the notebook using `cleaned_df.csv`.
   - **Option B – Start with the raw dataset** (full preprocessing):  
     Use `WELFake_Dataset.csv` and **uncomment the 4th code cell** to perform the data cleaning and preprocessing steps.

## Requirements
The notebook uses common Python data-science and machine-learning libraries such as:
- `pandas`
- `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`
- `seaborn`

Install any missing packages with your environment or IDE’s package manager (e.g., `pip install package_name`).

---
