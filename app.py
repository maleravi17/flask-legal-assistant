from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv

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
def initialize_gemini():
    """Initialize Gemini with the current API key."""
    global current_key_index
    try:
        genai.configure(api_key=API_KEYS[current_key_index])
        return genai.GenerativeModel('gemini-2.5-pro')
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
User Information: Name: Anil, Location: Mumbai, Maharashtra, Profession: Business Owner
User: What are the legal requirements for starting a small business?
Assistant: As a business owner in Mumbai, Maharashtra, you must comply with Indian and state-specific regulations. Key requirements include:
1. **Business Registration**: Register your business as a sole proprietorship, partnership, or private limited company under the Companies Act, 2013.
2. **GST Registration**: If your annual turnover exceeds ₹40 lakhs (₹20 lakhs for services), register for GST as per the CGST Act, 2017.
3. **Shops and Establishments Act**: Obtain a license from the Maharashtra Labour Department for your business premises in Mumbai.
4. **Local Permits**: Secure trade licenses from the Brihanmumbai Municipal Corporation (BMC) for operating in Mumbai.
5. **Tax Compliance**: Obtain a PAN and TAN for tax purposes.
Consult a local lawyer in Mumbai to ensure compliance with Maharashtra-specific regulations.

Example 2:
User Information: Name: Priya, Location: Chennai, Tamil Nadu, Profession: Student
User: Can a landlord evict a tenant without notice?
Assistant: As a student in Chennai, Tamil Nadu, understanding tenant rights is important. Under the Tamil Nadu Regulation of Rights and Responsibilities of Landlords and Tenants Act, 2017, a landlord cannot evict a tenant without notice. They must provide a written notice period (typically 30 days) and follow due process, such as obtaining a court order for eviction. For student accommodations, check your rental agreement for specific terms and consult a legal aid clinic in Chennai for affordable advice.

Example 3:
User Information: Name: Not provided, Location: Not provided, Profession: Not provided
User: Explain the process of filing a lawsuit in civil court.
Assistant: Here’s a general step-by-step process for filing a lawsuit in an Indian civil court:
1. **Consult a Lawyer**: Discuss your case to understand legal options.
2. **Draft the Complaint**: Prepare a document outlining your claims.
3. **File the Complaint**: Submit it to the appropriate civil court with the filing fee.
4. **Serve the Defendant**: Notify the defendant by serving the complaint.
5. **Await Response**: The defendant has time to respond.
6. **Discovery Phase**: Exchange evidence with the other party.
7. **Pre-Trial Motions**: File motions to resolve issues before trial.
8. **Trial**: Present your case before a judge.
9. **Judgment**: Receive the court’s decision.
10. **Appeal**: Appeal if needed.
Since your location and profession are not provided, consult a local lawyer for jurisdiction-specific guidance.
"""

# --- Endpoint ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_law_assistant(request: ChatRequest):
    global model

    # Validate inputs
    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="User query cannot be empty")

    # Combine provided user_history with current query
    session_data = [
        {"role": entry.role, "text": entry.text} for entry in request.user_history
    ]
    session_data.append({"role": "user", "text": request.user_query})

    # Create prompt
    prompt = f"""
        You are a legal bot with 20-25 years of experience as a lawyer, legal assistant, and attorney specializing in Indian law. Your goal is to provide accurate and helpful legal information tailored to the user's location and profession. If the location is provided (e.g., a city or state in India), incorporate relevant state-specific laws or local regulations. If the profession is provided (e.g., business owner, student, teacher), adjust the response to suit their context or needs. If location or profession is not provided, provide a general response applicable across India but mention that consulting a local lawyer is advisable for specific advice. Always include a disclaimer that your response is for educational purposes only and not a substitute for professional legal advice.

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

        return ChatResponse(response=assistant_response)
    except Exception as e:
        new_model = rotate_key()
        if new_model:
            model = new_model
            return await chat_with_law_assistant(request)
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Smart Law Assistant API"}

# --- Run the App ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))
