"""ML Gap Finder entry point.

Run the FastAPI application:
    uv run uvicorn src.main:app --reload

Or run directly:
    uv run python main.py
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
