from groq import Groq
from langchain_community.vectorstores.faiss import FAISS  # Facebook AI Similarity Search
from langchain_huggingface import HuggingFaceEmbeddings
import os


class MedicalChatbot:
    def __init__(self, api_key=None, model_name="llama3-70b-8192", temperature=0.7, max_chat_history_length=4, vec_db_path="../vec_db"):
        self.client = Groq(api_key="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") # api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        self.max_chat_history_length = max_chat_history_length
        self.vec_db_path = vec_db_path
        self.chat_history = []
        self.vec_db = self.load_vec_db()

    @staticmethod
    def generate_embeddings():
        """Initialize the embedding model."""
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_vec_db(self):
        """Load the FAISS vector database."""
        if os.path.exists(self.vec_db_path):
            print("Loading Vector Database...")
            return FAISS.load_local(folder_path=self.vec_db_path, embeddings=self.generate_embeddings(), allow_dangerous_deserialization=True)
        return None

    def retrieve_relevant_context(self, query, k=4):
        """Retrieve the most relevant documents from the FAISS vector database."""
        if self.vec_db:
            return self.vec_db.similarity_search(query=query, k=k)
        return None

    @staticmethod
    def get_sys_msg(context):
        """Generate a system message based on retrieved context."""
        if context:
            return f"You are a personal doctor Assistant who will answer questions about {context}."
        return None

    @staticmethod
    def format_context(context):
        """Format retrieved context for display."""
        return "\n\n".join([f'**Context {i+1}:**\n{doc.page_content}' for i, doc in enumerate(context)])

    def get_chat_response(self, prompt):
        """Generate a response using the LLM."""
        self.chat_history.append({"role": "user", "content": prompt})
        
        if len(self.chat_history) > self.max_chat_history_length:
            self.chat_history = self.chat_history[-self.max_chat_history_length:]
        
        context = self.retrieve_relevant_context(prompt, k=4)
        sys_msg = self.get_sys_msg(context)
        
        messages = [{"role": "system", "content": sys_msg}] + self.chat_history if sys_msg else self.chat_history
        
        response = ""
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
                    yield chunk.choices[0].delta.content  # Stream each chunk
            
            self.chat_history.append({"role": "assistant", "content": response})

        except Exception as e:
            # return f"❌ Error while generating response: {e}"
            yield f"❌ Error while generating response: {e}"
        
        # self.chat_history.append({"role": "assistant", "content": response})
        # return response

    def reset_chat_history(self):
        """Clear the chat history."""
        self.chat_history = []


# chatbot = MedicalChatbot(
#     api_key="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",
#     model_name="llama3-70b-8192",
#     temperature=0.7,
#     max_chat_history_length=4,
#     vec_db_path="/vec_db"
# )

# for chunk in chatbot.get_chat_response("What is the symptoms of Alzheimer disease?"):
#     print(chunk, end='')
