import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer 
import gradio as gr # Import Gradio

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

class FAQBot:
    """
    Intelligent FAQ Bot using Sentence Transformers for semantic similarity.
    """
    def __init__(self, csv_filepath):
        print("Loading FAQ data and model...")
        self.knowledge_base = self._load_data(csv_filepath)
        
        # Initialize the semantic model
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.question_embeddings = self._train_model()
        print(f"Bot initialized with {len(self.knowledge_base)} FAQs.")

    def _load_data(self, csv_filepath):
        """Loads and cleans the FAQ data, handling file path variations."""
        # Adjust file path based on common user setup/error context
        if not os.path.exists(csv_filepath) and os.path.exists("faqs.csv"):
            csv_filepath = "faqs.csv"
        
        try:
            df = pd.read_csv(csv_filepath, usecols=range(4))
            df.columns = ['Q#', 'Category', 'Question', 'Answer']
            df.dropna(subset=['Question', 'Answer'], inplace=True)
            return df[['Question', 'Answer']].to_dict('records')
        except Exception as e:
            print(f"Error loading data: {e}. Please ensure '{csv_filepath}' is correct.")
            return []

    def _train_model(self):
        """Encodes all FAQ questions into dense numerical vectors (embeddings)."""
        questions = [faq['Question'] for faq in self.knowledge_base]
        return self.model.encode(questions)

    def get_answer(self, user_question, confidence_threshold=0.6):
        """
        Finds the best matching answer using semantic similarity.
        Returns the answer text and the confidence score.
        """
        if not self.knowledge_base:
            return "My knowledge base is empty. I cannot answer any questions right now.", 0.0

        user_embedding = self.model.encode(user_question, convert_to_tensor=False)
        cosine_scores = cosine_similarity([user_embedding], self.question_embeddings)
        best_match_index = np.argmax(cosine_scores)
        best_score = cosine_scores[0, best_match_index]

        if best_score > confidence_threshold:
            best_faq = self.knowledge_base[best_match_index]
            return best_faq['Answer'], best_score
        else:
            fallback_message = "I'm sorry, I couldn't find a direct answer for that. Please try rephrasing or contact support."
            return fallback_message, best_score

# --- Main Bot Initialization ---
# Initialize the bot once globally
try:
    # Use the path based on your previous traceback, but gracefully fail to faqs.csv
    FAQ_BOT = FAQBot('assets/faqs.csv') 
except Exception as e:
    print(f"Failed to initialize the bot: {e}. Exiting.")
    exit()

# --- Gradio Interface Function ---

def respond(message, chat_history):
    """
    The function Gradio calls for every new message.
    """
    # Get the answer and the confidence score from the bot
    answer, score = FAQ_BOT.get_answer(message)
    
    # Optionally, include the confidence score in the response for transparency
    # formatted_answer = f"{answer} (Confidence: {score:.2f})"
    
    # Append the user message and bot response to the chat history
    chat_history.append((message, answer))
    
    # Return the updated chat history
    return "", chat_history

# --- Gradio UI Setup ---
with gr.Blocks(title="Intelligent FAQ Chatbot") as demo:
    gr.Markdown("# 🤖 Fashion Store - Sells apparel and bags")
    gr.Markdown(
        "Drop your fashion queries!"
    )

    # Chatbot component for displaying the conversation
    chatbot = gr.Chatbot(label="FAQ Chatbot", height=400)

    # Textbox for user input
    msg = gr.Textbox(
        label="Your Question", 
        placeholder="e.g., How do I cancel my order?", 
        scale=7
    )

    # Buttons for control
    with gr.Row():
        clear_btn = gr.Button("Clear Chat")
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    # Event Handlers: Define what happens when a button is clicked or Enter is pressed
    
    # When the user submits a message (either button or Enter)
    submit_btn.click(
        respond, # Function to call
        inputs=[msg, chatbot], # Inputs to the function
        outputs=[msg, chatbot] # Outputs to update
    )
    # Allows pressing Enter in the textbox to submit
    msg.submit(
        respond, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot]
    )

    # When the Clear button is clicked, clear the textbox and the chat history
    clear_btn.click(lambda: (None, []), outputs=[msg, chatbot])


# Launch the application
if __name__ == "__main__":
    demo.launch()