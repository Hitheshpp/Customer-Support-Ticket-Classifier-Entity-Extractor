
# 🛠️ Customer Support Ticket Classifier & Entity Extractor

This application classifies customer support tickets based on **issue type** and **urgency level**, and extracts relevant **entities** such as product names, complaint keywords, and dates using rule-based logic. A **Gradio interface** is provided for easy interaction.

## 🧰 Features
- Text cleaning, tokenization, lemmatization
- Feature engineering: TF-IDF, sentiment, ticket length, urgency score
- Multi-task classification (issue type, urgency level)
- Entity extraction: product names, complaint-related keywords, dates
- Web-based UI using Gradio

---

## 📂 Project Structure

```
project_folder/
│
├── ai_dev_assignment_tickets_complex_1000.xls   # Input dataset
├── ticket_classifier.py                         # All the code cells combined into a single Python file
├── README.md                                    # This file
```

---

## 🚀 How to Run

### 1. 🔧 Install Requirements

Make sure you have Python 3.7 or later. Then install the necessary libraries:

```bash
pip install pandas numpy scikit-learn nltk textblob gradio openpyxl
```

Also download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. 📁 Place Dataset

Put the file `ai_dev_assignment_tickets_complex_1000.xls` in the project directory or update the path in the code accordingly.

### 3. ▶️ Run the App

Execute the script using:

```bash
python ticket_classifier.py
```

Gradio will launch in your browser with an interface like:

```
Textbox Input:
> My laptop is not working, and I need a replacement urgently.

Output:
Issue: Hardware Problem
Urgency: High
Entities: {
  "products": ["Laptop"],
  "dates": [],
  "complaints": ["not working"]
}
```

---

## 🧠 Behind the Scenes

- **Text Preprocessing**: Lowercasing, punctuation removal, lemmatization, stopword removal.
- **Features**:
  - TF-IDF for content representation
  - Sentiment polarity using `TextBlob`
  - Urgency score based on predefined keywords
- **Models**:
  - `LogisticRegression` for issue classification
  - `RandomForestClassifier` for urgency level
- **Entities**:
  - Products: matched from known product list
  - Complaint keywords: rule-based
  - Dates: using regex

---

## 📦 Notes
- The product names are derived from the dataset column `product`.
- No external API is used, everything is handled offline.
- Sentiment and urgency keywords enhance prediction.

---

## ✅ Example Use Case
**Input**:  
```
My printer is broken and we need it fixed ASAP for an important meeting.
```

**Output**:  
```
Issue: Hardware Problem  
Urgency: High  
Entities:  
{
  "products": ["Printer"],
  "dates": [],
  "complaints": ["broken"]
}
```

---

## 📬 Contact
For questions or suggestions, please contact the project maintainer.
