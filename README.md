# Fake vs True News Classification / Classificazione delle Notizie False e Vere

This project involves classifying news stories as true or false based on their content. Using a dataset of fake and real news, we applied Natural Language Processing (NLP) and Machine Learning (ML) techniques to develop a model that will predict the truth value of a news story. Various data analysis steps, including word frequency analysis and sentiment analysis, were also used to analyze the dataset further.

Questo progetto prevede la classificazione delle notizie come vere o false in base al loro contenuto. Utilizzando un set di dati composto da notizie false e reali, abbiamo applicato tecniche di elaborazione del linguaggio naturale (NLP) e di apprendimento automatico (ML) per sviluppare un modello in grado di prevedere il valore di veridicità di una notizia. Per analizzare ulteriormente il set di dati sono state utilizzate anche varie fasi di analisi dei dati, tra cui l'analisi della frequenza delle parole e l'analisi del sentiment.

---

## Purpose of the Project / Scopo del Progetto

- **Categorize news articles as fake or true** based on text only.
- **Research the features** that are important in the classification (eg, keywords, number of words, sentiment).
- **Test the accuracy** of different machine learning models for text classification.
- Complete further **text analysis** to identify key attributes of fake and true news articles.

- **Classificare gli articoli di cronaca come falsi o veri** basandosi esclusivamente sul testo.
- **Ricercare le caratteristiche** importanti nella classificazione (ad esempio, parole chiave, numero di parole, sentiment).
- **Verificare l'accuratezza** dei diversi modelli di machine learning per la classificazione dei testi.
- Completare un'ulteriore **analisi dei testi** per identificare gli attributi chiave degli articoli di cronaca falsi e veri.

---

## Programs and Extensions Used / Programmi ed Estensioni Utilizzati

- **Python**: Used for data analysis, natural language processing and modeling.
- **Pandas** and **NumPy**: Used for data manipulation and analysis.
- **Scikit-learn**: All machine learning algorithms and model evaluation.
- **NLTK** and **spaCy**: Text preprocessing and other NLP tasks.
- **Matplotlib** and **Seaborn**: Data visualizations.
- **Visual Studio Code**: Used for code implementation and documentation/notes.

- **Python**: utilizzato per l'analisi dei dati, l'elaborazione del linguaggio naturale e la modellazione.
- **Pandas** e **NumPy**: utilizzati per la manipolazione e l'analisi dei dati.
- **Scikit-learn**: tutti gli algoritmi di apprendimento automatico e la valutazione dei modelli.
- **NLTK** e **spaCy**: pre-elaborazione del testo e altre attività di NLP.
- **Matplotlib** e **Seaborn**: visualizzazione dei dati.
- **Visual Studio Code**: utilizzato per l'implementazione del codice e la documentazione/le note.

---

## Data Set / Set di Dati

**Data source link**: [Fake News Detection Dataset - Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset consists of labeled news articles (fake vs real) with various text features. The data is pre-processed and cleaned to extract meaningful information for classification.

Il set di dati è costituito da articoli di cronaca etichettati (falsi vs reali) con varie caratteristiche testuali. I dati sono pre-elaborati e puliti per estrarre informazioni significative ai fini della classificazione.

---

## Analysis / Analisi

- **Exploratory Data Analysis (EDA)**: Taking a look at your dataset to get an overview of its structure and major characteristics, like word frequency, article length, and more.
- **Text Pre-processing:** Cleaning strings of text and tokenizing the text data with NLTK and spaCy libraries.
- **Model Training:** Creating and assessing machine learning models (Logistic Regression, Naive Bayes, and Random Forest) to classify news articles.
- **Feature Engineering:** Investigating and selecting the best features to improve classification performance.

- **Analisi esplorativa dei dati (EDA)**: esame del set di dati per ottenere una panoramica della sua struttura e delle sue caratteristiche principali, come la frequenza delle parole, la lunghezza degli articoli e altro ancora.
- **Pre-elaborazione del testo:** pulizia delle stringhe di testo e tokenizzazione dei dati testuali con le librerie NLTK e spaCy.
- **Addestramento del modello:** creazione e valutazione di modelli di apprendimento automatico (regressione logistica, Naive Bayes e Random Forest) per classificare gli articoli di cronaca.
- **Feature engineering:** ricerca e selezione delle caratteristiche migliori per migliorare le prestazioni di classificazione.

---

## Results & Analysis / Risultati e Analisi

> Below are the key findings and model evaluations for the Fake vs True News project.
> Di seguito sono riportati i risultati principali e le valutazioni del modello per il progetto Fake vs True News.

---

### Key Findings / Risultati Principali

- **Article Length**: Articles that are fake news tend to be shorter than true news.
  - **Lunghezza dell'articolo**: Gli articoli di notizie false tendono a essere più brevi di quelli di notizie vere.

- **Word Frequency**: Certain words appear more frequently in fake news than in true news, such as "falso," [illegible], and "cospirazioni."
  - **Frequenza delle parole**: Nelle notizie false si trovano piú frequentemente certe parole come falso, scandalo, cospirazione.

- **Model Performance**: The logistic Regression model had the best performance with an accuracy of 92%, followed by Random forest model at 89%.

- **Lunghezza dell'articolo**: gli articoli che contengono notizie false tendono ad essere più brevi rispetto a quelli che riportano notizie vere.
  - **Frequenza delle parole**: alcune parole compaiono più frequentemente nelle notizie false rispetto a quelle vere, come “falso”, [illeggibile] e “cospirazioni”.

- **Frequenza delle parole**: alcune parole compaiono più frequentemente nelle notizie false che in quelle vere, come “falso”, [illeggibile] e “cospirazioni”.
  - **Frequenza delle parole**: alcune parole compaiono più frequentemente nelle notizie false che in quelle vere, come “falso”, [illeggibile] e “cospirazioni”.

- **Prestazioni del modello**: il modello di regressione logistica ha ottenuto le migliori prestazioni con un'accuratezza del 92%, seguito dal modello random forest con l'89%.

---

### 1. Exploratory Data Analysis (EDA) / Analisi Esplorativa dei Dati (EDA)

I reviewed the main characteristics of the dataset, including the imbalance of fake news versus true news, article lengths, and frequencies of common words.

- **Imbalance of Fake vs True news**: The dataset has approximately **50%** fake news and **50%** true news articles.
- **Article Length**: True news articles were longer, on average, than fake news articles.
- **Most Common Words**: Fake news articles had greater frequencies of words such as "scandal" , "breaking", and "exclusive."

Ho esaminato le caratteristiche principali del set di dati, tra cui lo squilibrio tra notizie false e notizie vere, la lunghezza degli articoli e la frequenza delle parole comuni.

- **Squilibrio tra notizie false e notizie vere**: il set di dati contiene circa il **50%** di notizie false e il **50%** di notizie vere.
- **Lunghezza degli articoli**: gli articoli di notizie vere erano in media più lunghi di quelli di notizie false.
- **Parole più comuni**: gli articoli di notizie false presentavano una maggiore frequenza di parole come “scandalo”, “ultime notizie” ed “esclusiva”.

---

### 2. Text Pre-processing and Feature Engineering / Pre-elaborazione del Testo e Feature Engineering

The text data was pre-processed with **tokenization**, **stop-word removal**, and **stemming**. We calculated key metrics for the most significant words in the articles such as **TF-IDF (Term Frequency-Inverse Document Frequency)** score.

- **Wordcloud for Frequent Terms** - A wordcloud depicting the most frequent terms was produced from both the true articles and fake news articles. 
- **Feature Selection** - Certain words including "conspiracy", "government", and "news" were noted to have a strong impact on fake news articles.

I dati testuali sono stati pre-elaborati con **tokenizzazione**, **rimozione delle parole vuote** e **stemming**. Abbiamo calcolato le metriche chiave per le parole più significative negli articoli, come il punteggio **TF-IDF (Term Frequency-Inverse Document Frequency)**.

- **Wordcloud per i termini frequenti** - È stato creato un wordcloud che raffigura i termini più frequenti sia negli articoli veri che in quelli falsi.
- **Selezione delle caratteristiche** - Alcune parole, tra cui “cospirazione”, ‘governo’ e “notizie”, hanno avuto un forte impatto sugli articoli falsi.

#### Word Frequency Comparison / Confronto Frequenza Parole

- **Wordcloud of Fake News**: Certain words appeared frequently in the fake news articles, such as "scandal," "breaking," "exclusive."
- **Wordcloud of True News**: The true news articles had words like "report," "confirmed," and "official."

- **Wordcloud delle fake news**: alcune parole ricorrevano frequentemente negli articoli di fake news, come “scandalo”, “ultime notizie” ed “esclusiva”.
- **Wordcloud delle notizie vere**: gli articoli di notizie vere contenevano parole come “rapporto”, ‘confermato’ e “ufficiale”.

#### Model Performance / Prestazioni del Modello

- **Confusion Matrix** from Logistic Regression: Precision, recall are high for both fake and true news.
- **ROC Curve**: The Logistic Regression model has high AUC (Area Under Curve) level **0.95**, indicating a great performance of the model.

- **Matrice di confusione** dalla regressione logistica: precisione e richiamo sono elevati sia per le notizie false che per quelle vere.
- **ROC Curve**: il modello di regressione logistica ha un elevato livello di AUC (area sotto la curva) pari a **0,95**, che indica un ottimo rendimento del modello.


---

### 3. Model Evaluation / Valutazione del Modello

We compared the performance of several machine learning models:
Ho confrontato le prestazioni di diversi modelli di apprendimento automatico:

| Model               | Accuracy | Precision (Fake) | Recall (Fake) | F1 Score (Fake) |
|---------------------|----------|------------------|----------------|-----------------|
| Logistic Regression  | 0.92     | 0.89             | 0.91           | 0.90            |
| Naive Bayes         | 0.87     | 0.84             | 0.80           | 0.82            |
| Random Forest       | 0.89     | 0.86             | 0.88           | 0.87            |

---

## 4. Model Visualizations / Visualizzazioni dei Modelli

> Below are some visualizations of the models used in the project. 
> Di seguito sono riportate alcune visualizzazioni dei modelli utilizzati nel progetto.

### 4.1 Logistic Regression Model

![Logistic Regression](./plots/LogisticRegression.png)

> This is the visualization of the **logistic regression** model, showing its performance in classifying fake and real news.
> Questa è la visualizzazione del modello di **regressione logistica**, che mostra le sue prestazioni nella classificazione delle notizie false e reali.

### 4.2 Naive Bayes Model

![Naive Bayes](./plots/naive_bayes.png)

> This is a visualization of the **Naive Bayes** model, demonstrating its accuracy and ability to distinguish between fake and real news.
> Questa è la visualizzazione del modello **Naive Bayes**, che dimostra la sua accuratezza e capacità di distinguere tra notizie false e reali.

### 4.3 SVM Confusion Matrix

![SVM Confusion Matrix](./plots/confusion_matrix_SVM.png)

> The **SVM (Support Vector Machine)** confusion matrix reveals how well the model is able to distinguish between fake and real news articles, with high accuracy and recall.
> La matrice di confusione **SVM (Support Vector Machine)** rivela quanto il modello sia in grado di distinguere tra articoli di cronaca falsi e reali, con elevata precisione e richiamo.

### 4.4 TF-IDF Confusion Matrix

![TF-IDF Confusion Matrix](./plots/TF_IDF_confusion_matrix.png)

> The confusion matrix for the vectorized **TF-IDF** model shows how it classifies fake and real news, with balanced performance in distinguishing between the two categories.
> La matrice di confusione per il modello vettorializzato **TF-IDF** mostra come classifica le notizie false e quelle vere, con prestazioni equilibrate nella distinzione tra le due categorie.

---

### 5. Discussion / Discussione

- **Logistic Regression** outperformed the other models, providing the highest accuracy and recall, and made the best job at identifying fake news articles.
- **Naive Bayes** had good performance and was quite close in terms of recall, but was less effective than Logistic Regression.
- Feature engineering (TF-IDF, word frequency) really helped with the model's classification abilities.

- La **regressione logistica** ha superato gli altri modelli, fornendo la massima accuratezza e richiamo, e ha ottenuto i migliori risultati nell'identificazione di articoli di fake news.
- Il modello **Naive Bayes** ha ottenuto buone prestazioni ed era abbastanza simile in termini di richiamo, ma era meno efficace della regressione logistica.
- Feature engineering (TF-IDF, frequenza delle parole) ha davvero aiutato le capacità di classificazione del modello.

---

### 6. Limitations and Future Work / Limiti e Lavoro Futuro

- **Data Bias**: There may be bias in the dataset, especially with language or topic. A more varied dataset may help with generalization of the models.
- **Complex Models**: Future iterations could include complex models like **XGBoost** or **BERT** (Bidirectional Encoder Representations from Transformers) that may improve performance.
- **Sentiment Analysis**: Adding sentiment analysis could offer additional features that might improve the accuracy of classification.

- **Distorsione dei dati**: potrebbero esserci distorsioni nel set di dati, specialmente per quanto riguarda la lingua o l'argomento. Un set di dati più vario potrebbe aiutare nella generalizzazione dei modelli.
- **Modelli complessi**: le iterazioni future potrebbero includere modelli complessi come **XGBoost** o **BERT** (Bidirectional Encoder Representations from Transformers) che potrebbero migliorare le prestazioni.
- **Analisi del sentiment**: l'aggiunta dell'analisi del sentiment potrebbe offrire funzionalità aggiuntive in grado di migliorare l'accuratezza della classificazione.

---

## Ethics and Data Privacy / Etica e Privacy dei Dati

The data utilized in this project is publicly available data, and everything was done to make sure ethical considerations are satisfied. There was no personal information or sensitive information used or exposed during the analysis.

I dati utilizzati in questo progetto sono dati disponibili pubblicamente ed è stato fatto tutto il possibile per garantire il rispetto delle considerazioni etiche. Durante l'analisi non sono state utilizzate né esposte informazioni personali o sensibili.
---


