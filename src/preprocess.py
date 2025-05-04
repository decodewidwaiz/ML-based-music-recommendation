import pandas as pd
import re
import joblib
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# Ensure NLTK resources are available
required_resources = {
    "punkt": "tokenizers/punkt",
    "stopwords": "corpora/stopwords"
}

for name, path in required_resources.items():
    try:
        nltk.data.find(path)
        logging.info(f"✅ NLTK resource '{name}' already available.")
    except LookupError:
        logging.info(f"⬇️ Downloading NLTK resource: {name}")
        nltk.download(name)

# Load and sample dataset
try:
    df = pd.read_csv("spotify_millsongdata.csv").sample(10000)
    logging.info("✅ Dataset loaded and sampled: %d rows", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop 'link' column if it exists
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Prepare stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    try:
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        tokens = word_tokenize(text, language='english')
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        logging.warning(f"⚠️ Error processing text: {text[:30]}... | Error: {e}")
        return ""

# Apply preprocessing
logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# TF-IDF vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Cosine similarity computation
logging.info("📐 Calculating cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
logging.info("✅ Cosine similarity matrix generated.")

# Save results
joblib.dump(df, 'df_cleaned.pkl')
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
joblib.dump(cosine_sim, 'cosine_sim.pkl')
logging.info("💾 Data saved to disk.")

logging.info("✅ Preprocessing complete.")
