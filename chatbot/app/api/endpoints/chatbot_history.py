from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse
import yaml
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Initialize API router for FastAPI
router = APIRouter()

# ---------------------------
# Configuration Loading
# ---------------------------
# Load configuration settings from 'config.yaml'
with open('config.yaml', 'r') as configfile:
    config = yaml.safe_load(configfile)
    db_params = config['db_params']  # Database connection parameters

# ---------------------------
# Connect to PostgreSQL Database
# ---------------------------
# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

# ---------------------------
# Helper Functions
# ---------------------------

def get_chat_history(token, order='asc'):
    """
    Retrieve the chat history for a given discussion token.
    
    Args:
        token (str): Unique token identifying the discussion.
        order (str, optional): Order of the history ('asc' for ascending, 'desc' for descending). Defaults to 'asc'.
    
    Returns:
        list: Formatted list of conversation history with questions and answers.
    
    Raises:
        HTTPException: If the discussion is not found or a database error occurs.
    """
    try:
        # Fetch the discussion ID based on the provided token
        cursor.execute("SELECT id FROM discussions WHERE token = %s", (token,))
        result = cursor.fetchone()
        if not result:
            # Raise a 404 error if the discussion token is not found
            raise HTTPException(status_code=404, detail="Discussion not found")
        
        discussion_id = result[0]

        # Retrieve all messages related to the discussion, ordered by timestamp ascending
        cursor.execute("""
            SELECT role, message FROM discussion_messages
            WHERE discussion_id = %s ORDER BY timestamp ASC
        """, (discussion_id,))
        
        history_rows = cursor.fetchall()
        
        # Format the fetched messages into a structured history list
        history = []
        current_entry = None
        for row in history_rows:
            role, message = row
            if role == "user":
                # If a new user message is found, append the previous entry if it exists
                if current_entry:
                    history.append(current_entry)
                # Start a new history entry for the user's question
                current_entry = {"question": message, "answer": ""}
            elif role == "assistant" and current_entry:
                # Assign the assistant's answer to the current history entry
                current_entry["answer"] = message
        
        # Append the last user entry if it exists
        if current_entry:
            history.append(current_entry)
        
        # Reverse the history if the requested order is descending
        if order == 'desc':
            history.reverse()
        
        return history
    except psycopg2.Error as db_error:
        # Rollback the transaction in case of a database error
        conn.rollback()
        # Raise a 500 Internal Server Error with the database error details
        raise HTTPException(status_code=500, detail="Database error occurred: " + str(db_error))

# ---------------------------
# API Endpoint
# ---------------------------

@router.get("/history")
async def chat_history(
    token: str = Query(None, description="Unique token for discussion"),
    order: str = Query('asc', regex='^(asc|desc)$', description="Order of history: 'asc' or 'desc'")
):
    """
    Endpoint to retrieve the chat history for a given discussion token.
    
    Args:
        token (str): Unique token identifying the discussion.
        order (str, optional): Order of the history ('asc' or 'desc'). Defaults to 'asc'.
    
    Returns:
        JSONResponse: JSON object containing the formatted chat history or error details.
    """
    try:
        # Validate that the token is provided
        if not token:
            # Raise a 400 Bad Request error if the token is missing
            raise HTTPException(status_code=400, detail="Token is required")

        # Retrieve the chat history using the helper function
        history = get_chat_history(token, order)
        # Return the history in a JSON response
        return JSONResponse(content={"history": history})

    except HTTPException as e:
        # Handle known HTTP exceptions and return their details
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code)
    except psycopg2.Error as db_error:
        # Handle database-specific errors by rolling back and returning a 500 error
        conn.rollback()
        return JSONResponse(content={"detail": "Database error occurred: " + str(db_error)}, status_code=500)
    except Exception as e:
        # Handle any unexpected exceptions
        print(f"Unexpected error: {e}")  # Optional: Log the exception details for debugging
        # Rollback the transaction for unexpected errors as well
        conn.rollback()
        # Return a generic 500 Internal Server Error message
        return JSONResponse(content={"detail": "Internal server error"}, status_code=500)
