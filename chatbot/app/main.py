from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import chatbot
from app.api.endpoints import chatbot_history


app = FastAPI()

# Define allowed origins for CORS
origins = [
    "http://193.94.15.116",  # Example: allow localhost
    "https://193.94.15.116",  # Example: allow localhost on a specific port
    "*",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins to access the API
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(chatbot.router, prefix="/api/v1/chatbot")
app.include_router(chatbot_history.router, prefix="/api/v1/chatbot")



@app.get("/")
async def read_root():
    return {"message": "Welcome to the Chatbot API"}
