from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Load API keys from .env
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

# Initialize the current key index
current_key_index = 0

# Folder to save session data
SESSION_FOLDER = "sessions"
os.makedirs(SESSION_FOLDER, exist_ok=True)

# --- Helper Functions ---
def load_session(session_id: str) -> List[Dict]:
    """Load session data from a file."""
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            return json.load(f)
    return []

def save_session(session_id: str, session_data: List[Dict]):
    """Save session data to a file."""
    session_file = os.path.join(SESSION_FOLDER, f"{session_id}.json")
    with open(session_file, "w") as f:
        json.dump(session_data, f)

def initialize_gemini():
    """Initialize Gemini with the current API key."""
    global current_key_index
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        return genai.GenerativeModel('gemini-2.0-flash-001')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Gemini: {e}")

def rotate_key():
    """Rotate to the next API key."""
    global current_key_index
    if current_key_index < len(API_KEYS) - 1:
        current_key_index += 1
        return initialize_gemini()
    else:
        raise HTTPException(status_code=500, detail="All API keys have been used. Please add more keys.")

# Initialize Gemini model
model = initialize_gemini()

# --- FastAPI App ---
app = FastAPI(title="Smart Law Assistant API")

# Configure CORS for front-end communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class UserInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None 
    profession: Optional[str] = None 

class ConversationEntry(BaseModel):
    role: str 
    text: str

class ChatRequest(BaseModel):
    session_id: str
    user_query: str
    user_info: UserInfo
    user_history: List[ConversationEntry]

class ChatResponse(BaseModel):
    response: str
    disclaimer: str = "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."

# --- Two-shot Prompting Examples ---
examples = """
Example 1:
User: What is the difference between civil law and criminal law?
Assistant: Civil law deals with disputes between individuals or organizations, such as contracts or property disputes. Criminal law, on the other hand, involves actions that are harmful to society and are prosecuted by the state, such as theft or assault.

Example 2:
User: Can a lawyer represent both parties in a case?
Assistant: No, a lawyer cannot represent both parties in a case due to a conflict of interest. It is unethical and prohibited by legal professional standards.

Example 3:
User: Explain the process of filing a lawsuit in civil court.
Assistant: Sure! Here's a step-by-step explanation:
1. **Consult a Lawyer**: Discuss your case with a lawyer to understand your legal options.
2. **Draft the Complaint**: Prepare a legal document outlining your claims and the relief you seek.
3. **File the Complaint**: Submit the complaint to the appropriate court and pay the filing fee.
4. **Serve the Defendant**: Notify the defendant about the lawsuit by serving them the complaint.
5. **Await Response**: The defendant has a specified time to respond to the complaint.
6. **Discovery Phase**: Both parties exchange information and evidence related to the case.
7. **Pre-Trial Motions**: Either party can file motions to resolve the case before trial.
8. **Trial**: If the case proceeds to trial, both parties present their arguments and evidence.
9. **Judgment**: The judge or jury delivers a verdict.
10. **Appeal**: If either party is dissatisfied, they can appeal the decision.
"""

# --- Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_law_assistant(request: ChatRequest):
    global model  # Declare 'model' as global

    # Validate inputs
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="User query cannot be empty")

    # Load session data
    session_data = load_session(request.session_id)

    # Combine provided user_history with session data to maintain conversation continuity
    session_data.extend([
        {"role": entry.role, "text": entry.text} for entry in request.user_history
    ])

    # Add the current user query to session history
    session_data.append({"role": "user", "text": request.user_query})

    # Create a context-specific prompt
    prompt = f"""
        You are a legal bot with 20-25 years of experience as a lawyer, legal assistant, and attorney specializing in Indian law, Bharatiya Nyaya Sanhita, 2023 (replacing the Indian Penal Code), Bharatiya Nagarik Suraksha Sanhita, 2023 (replacing the Code of Criminal Procedure), Bharatiya Sakshya Adhiniyam, 2023 (replacing the Indian Evidence Act), Indian Acts, court judgments, passport-related issues, government-related official work, and various branches of Indian law such as corporate law, family law, criminal law, constitutional law, labor law, property law, and other relevant legal domains. Your role is to determine legal rights and provide accurate, professional guidance to legal questions, including relevant IPC section numbers, Indian Acts, case law citations, and specific advice tailored to the applicable branch of law.
        Follow these guidelines:
            * Provide answers in plain, easy-to-understand language.
            * Include the disclaimer: "Disclaimer: This information is for educational purposes only and should not be considered legal advice. It is essential to consult with a legal professional for specific guidance regarding your situation."
            * If the user asks a question in a local language, respond in the same language to assist them effectively.
            * Provide source websites or URLs to support your answers where applicable (e.g., official government sites, legal databases).
            * For specific legal precedents or case law, include relevant citations (e.g., case name, court, and year) with a brief summary of the judgment.
            * Format responses with clear paragraphs separated by double newlines and use bullet points (e.g., '* ') for lists or key points.
            * If the question is vague, overly broad, or lacks specific details, provide a concise, general response based on a reasonable interpretation of the user’s intent. Then, follow up with intelligent, context-aware questions to clarify the user’s needs.
            * If the user references or uses provisions from the old laws (Indian Penal Code, Code of Criminal Procedure, or Indian Evidence Act), provide the answer based on the corresponding provisions of the new laws (Bharatiya Nyaya Sanhita, 2023, Bharatiya Nagarik Suraksha Sanhita, 2023, or Bharatiya Sakshya Adhiniyam, 2023) and, where relevant, clarify the transition or differences.
            * If the question is not related to Indian law, IPC sections, Indian Acts, judgments, government-related work, or any branch of Indian law (e.g., corporate law, family law), politely decline to answer, stating: "I'm sorry, but I can only assist with questions related to Indian law and legal matters. Please ask a legal question, and I'll be happy to help."
        Use the conversation history to maintain context, especially for follow-up questions. If a question is unclear, refer to the history to infer context and tailor clarification questions accordingly.
        Tailor responses to the user's profession where relevant to provide context-specific legal guidance.
        User Information:
        Name: {request.user_info.name or 'Not provided'}
        Location: {request.user_info.location or 'Not provided'}
        Profession: {request.user_info.profession or 'Not provided'}

    {examples}

    Conversation History:
    {" ".join([f"{msg['role']}: {msg['text']}" for msg in session_data])}

    User: {request.user_query}
    Assistant:
    """

    try:
        response = model.generate_content(prompt)
        assistant_response = response.text

        # Add the assistant's response to the session history
        session_data.append({"role": "assistant", "text": assistant_response})

        # Save the updated session data
        save_session(request.session_id, session_data)

        return ChatResponse(response=assistant_response)
    except Exception as e:
        # Rotate 
        new_model = rotate_key()
        if new_model:
            model = new_model  
            return await chat_with_law_assistant(request)
        else:
            raise HTTPException(status_code=500, detail="Sorry, I am unable to process your request at the moment.")

@app.get("/")
async def root():
    return {"message": "Welcome to Smart Law Assistant API"}

# --- Run the App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
