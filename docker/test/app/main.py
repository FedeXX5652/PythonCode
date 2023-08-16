from typing import Union
import uvicorn
from fastapi import FastAPI

app = FastAPI(title = "FastAPI test with Docker",
            description = "This is a test of FastAPI with Docker",
            version = "0.0.1", contact = {"name": "FedeXX", "email": "fedegalant@gmail.com"})


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, loop="asyncio")