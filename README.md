
# Movie Recommendation System

## 1. Project
Movie Recommendation System Using Machine Learning

## 2. Technique
Content-Based Filtering

## 3. Main methods
- TF-IDF Vectorization
- Cosine Similarity

## 4. Dataset
Hindi Bollywood movie dataset from the public
[Bollywood Movie Dataset](https://github.com/calci/bollywood-movie-dataset).
The app downloads the dataset automatically and uses its Hindi movie titles,
genres, and release years.

## 5. Features
- Movie title
- Movie genres
- Year extracted from title

## 6. How to run

### Step 1: Install Python
Use Python 3.10 or newer.

### Step 2: Open terminal in this project folder

### Step 3: Install libraries
```bash
pip install -r requirements.txt
```

### Step 4: Run
```bash
streamlit run app.py
```

### Step 5:
A browser window will open. Select a movie and click "Recommend Movies".

## 7. Movie posters
The app works without a poster API key and displays placeholders.

For real posters:
1. Create a free TMDB account.
2. Get a TMDB API key.
3. Set an environment variable named `TMDB_API_KEY`.

Windows PowerShell:
```powershell
$env:TMDB_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

Windows Command Prompt:
```cmd
set TMDB_API_KEY=YOUR_KEY_HERE
streamlit run app.py
```

The poster API is optional. The ML recommendation system does not depend on it.

## 8. Project flow

Hindi Bollywood Movie Dataset
        ↓
Data Preprocessing
        ↓
Movie Genres
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Find Similar Movies
        ↓
Display Recommendations

## 9. Important viva answer

"My project uses Content-Based Filtering. TF-IDF converts movie genre text into numerical vectors, and Cosine Similarity calculates similarity between movies. The system recommends movies with the highest similarity scores."
