from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import engine, get_db
from models import Base, UserDetails, UsersChat, UserChatDetails

# Create database tables
Base.metadata.create_all(bind=engine)

# Load environment variables
load_dotenv()

# --- Configuration ---
API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]
current_key_index = 0

# --- Helper Functions ---
def load_session(session_id: str, db: Session) -> List[Dict]:
    """Load session data from the database."""
    chat = db.query(UsersChat).filter(UsersChat.id == int(session_id), UsersChat.is_deleted == False).first()
    if not chat:
        return []
    
    chat_details = db.query(UserChatDetails).filter(
        UserChatDetails.chat_id == int(session_id),
        UserChatDetails.is_deleted == False
    ).order_by(UserChatDetails.created.asc()).all()
    
    session_data = []
    for detail in chat_details:
        if detail.question:
            session_data.append({"role": "user", "text": detail.question})
        if detail.answer:
            session_data.append({"role": "assistant", "text": detail.answer})
    return session_data

def save_session(session_id: str, user_id: int, session_data: List[Dict], db: Session):
    """Save session data to the database."""
    # Check if chat session exists
    chat = db.query(UsersChat).filter(UsersChat.id == int(session_id), UsersChat.is_deleted == False).first()
    if not chat:
        # Create new chat session
        chat = UsersChat(
            id=int(session_id),
            chat_name=f"Chat {session_id}",
            user_id=user_id,
            is_deleted=False
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)

    # Save latest question and answer
    latest_entry = session_data[-1] if session_data else None
    if latest_entry and latest_entry["role"] == "assistant":
        second_last = session_data[-2] if len(session_data) > 1 else None
        if second_last and second_last["role"] == "user":
            chat_detail = UserChatDetails(
                chat_id=int(session_id),
                question=second_last["text"],
                answer=latest_entry["text"],
                is_deleted=False
            )
            db.add(chat_detail)
            db.commit()

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

# Configure CORS
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
    user_id: int
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
async def chat_with_law_assistant(request: ChatRequest, db: Session = Depends(get_db)):
    global model

    # Validate inputs
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="User query cannot be empty")

    # Verify user exists
    user = db.query(UserDetails).filter(UserDetails.id == request.user_id, UserDetails.is_deleted == False).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Load session data
    session_data = load_session(request.session_id, db)

    # Combine provided user_history with session data
    session_data.extend([
        {"role": entry.role, "text": entry.text} for entry in request.user_history
    ])

    # Add current user query to session history
    session_data.append({"role": "user", "text": request.user_query})

    # Create prompt
    prompt = f"""
        You are a legal bot with 20-25 years of experience as a lawyer, legal assistant, and attorney specializing in Indian law...
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

        # Add assistant's response to session history
        session_data.append({"role": "assistant", "text": assistant_response})

        # Save updated session data
        save_session(request.session_id, request.user_id, session_data, db)

        return ChatResponse(response=assistant_response)
    except Exception as e:
        new_model = rotate_key()
        if new_model:
            model = new_model
            return await chat_with_law_assistant(request, db)
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Smart Law Assistant API"}

# --- Run the App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))
