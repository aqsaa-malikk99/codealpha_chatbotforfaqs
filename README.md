# CodeAlpha_ChatbotforFAQs

## 🤖 Intelligent FAQ Chatbot (CodeAlpha AI Internship - Task 2)

This project delivers an intelligent and user-friendly Chatbot designed to answer Frequently Asked Questions (FAQs) for a hypothetical fashion and apparel e-commerce store.

The key feature of this chatbot is its reliance on **Semantic Similarity**, which allows it to understand the underlying _meaning_ of a user's question, even if the phrasing is different from the original FAQ. This solves the limitation of traditional keyword-matching bots.

A clean, interactive web interface is provided using the **Gradio** framework.

### ✅ CodeAlpha Task 2 Requirements Met

|

Requirement

|

Status

|

Implementation Details

|
|

**Collect FAQs**

|

**Complete**

|

Data is sourced from the included `faqs.csv` file.

|
|

**Preprocess Text**

|

**Complete**

|

**Sentence Embeddings** are generated using `sentence-transformers`, which handles tokenization and vector representation.

|
|

**Match User Questions**

|

**Complete**

|

**Semantic Matching** is performed via **Cosine Similarity** on the dense vectors created by the pre-trained `all-MiniLM-L6-v2` model.

|
|

**Display Answer**

|

**Complete**

|

The best-matching FAQ answer is presented as the bot's response.

|
|

**Optional: Simple Chat UI**

|

**Complete**

|

A dynamic web interface is built using **Gradio** for real-time interaction.

|

### 🧠 Technical Intelligence (How It Works)

The core intelligence is contained within the `FAQBot` class:

1.  **Model Initialization:** The bot uses the `SentenceTransformer('all-MiniLM-L6-v2')` model to load a pre-trained language understanding model.

2.  **Embedding Generation:** All questions from `faqs.csv` are encoded into **dense numerical vectors (embeddings)** that capture the sentence's meaning.

3.  **Real-Time Query:** When a user asks a question, it is also converted into a semantic vector.

4.  **Similarity Calculation:** `sklearn.metrics.pairwise.cosine_similarity` is used to find the vector closest to the user's query. The question with the smallest angular distance (highest cosine score) is determined to be the semantic match, ensuring high accuracy for paraphrased queries like:

    - **User Query:** _"Can I return my shoes?"_

    - **Bot Match:** _"What is your return policy?"_

### 🛠️ Technology Stack

- **Language:** Python

- **Data Processing:** `pandas`

- **NLP/Intelligence:** `sentence-transformers`

- **Similarity Metric:** `scikit-learn` (`cosine_similarity`)

- **User Interface:** `gradio`

### 🚀 Setup and Run Instructions

To run the chatbot, you need Python 3.8+ and the project files: `faq_ui.py`, `faqs.csv`, and `requirements.txt`.

**1\. Clone the Repository:**

```
git clone [https://github.com/YourUsername/CodeAlpha_ChatbotforFAQs.git](https://github.com/YourUsername/CodeAlpha_ChatbotforFAQs.git)
cd CodeAlpha_ChatbotforFAQs

```

**2\. Install Dependencies:**

The following libraries are required. We recommend using a virtual environment.

**`requirements.txt` Content:**

```
pandas
numpy
scikit-learn
sentence-transformers
gradio

```

**Installation Command:**

```
pip install -r requirements.txt

```

**3\. Run the Chatbot:**

Execute the main Python script. The Gradio library will launch the web server.

```
python faq_ui.py

```

The terminal will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this link in your browser to interact with the bot.
