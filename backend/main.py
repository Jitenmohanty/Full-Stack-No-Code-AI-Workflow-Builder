from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Dict
import json

from database import get_db, Document, Workflow, ChatHistory
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.workflow_executor import WorkflowExecutor

app = FastAPI(title="AI Workflow Builder API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
doc_processor = DocumentProcessor()
embedding_service = EmbeddingService()
workflow_executor = WorkflowExecutor()

@app.get("/")
def read_root():
    return {"message": "AI Workflow Builder API", "version": "1.0.0"}

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = "default",
    embedding_model: str = "openai",
    db: Session = Depends(get_db)
):
    """Upload and process a document"""
    try:
        # Read file
        file_bytes = await file.read()
        
        # Process document
        result = doc_processor.process_document(file_bytes, file.filename)
        
        # Store in database
        doc = Document(
            filename=result["filename"],
            content=result["text"],
            metadata={"chunk_count": result["chunk_count"]}
        )
        db.add(doc)
        db.commit()
        
        # Add to vector store
        embedding_result = embedding_service.add_documents(
            collection_name=collection_name,
            chunks=result["chunks"],
            embedding_model=embedding_model
        )
        
        return {
            "status": "success",
            "document_id": doc.id,
            "filename": file.filename,
            "chunks": result["chunk_count"],
            "embedding_result": embedding_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflows/save")
async def save_workflow(
    workflow_data: Dict,
    db: Session = Depends(get_db)
):
    """Save a workflow"""
    workflow = Workflow(
        name=workflow_data.get("name", "Untitled Workflow"),
        nodes=workflow_data.get("nodes", []),
        edges=workflow_data.get("edges", []),
        config=workflow_data.get("config", {})
    )
    db.add(workflow)
    db.commit()
    
    return {"status": "success", "workflow_id": workflow.id}

@app.get("/api/workflows/{workflow_id}")
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get a workflow by ID"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "nodes": workflow.nodes,
        "edges": workflow.edges,
        "config": workflow.config
    }

@app.post("/api/workflows/execute")
async def execute_workflow(
    execution_data: Dict,
    db: Session = Depends(get_db)
):
    """Execute a workflow"""
    try:
        result = await workflow_executor.execute_workflow(
            query=execution_data["query"],
            nodes=execution_data["nodes"],
            edges=execution_data["edges"],
            node_configs=execution_data.get("nodeConfigs", {})
        )
        
        # Save chat history
        if execution_data.get("workflow_id"):
            chat_user = ChatHistory(
                workflow_id=execution_data["workflow_id"],
                role="user",
                content=execution_data["query"]
            )
            chat_assistant = ChatHistory(
                workflow_id=execution_data["workflow_id"],
                role="assistant",
                content=result["response"],
                metadata=result.get("metadata", {})
            )
            db.add(chat_user)
            db.add(chat_assistant)
            db.commit()
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{workflow_id}")
async def get_chat_history(workflow_id: int, db: Session = Depends(get_db)):
    """Get chat history for a workflow"""
    history = db.query(ChatHistory).filter(
        ChatHistory.workflow_id == workflow_id
    ).order_by(ChatHistory.created_at).all()
    
    return [{
        "role": msg.role,
        "content": msg.content,
        "created_at": msg.created_at.isoformat()
    } for msg in history]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)