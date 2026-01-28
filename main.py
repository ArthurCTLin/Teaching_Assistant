from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from app.schemas import MathAnalysisSchema, AnalysisRequest
from app.services import MathAIService
from app.services.prompts import SAT_MATH_ANALYZER_PROMPT, SIMILAR_QUESTION_PROMPT
from pathlib import Path
import os
import shutil

app = FastAPI(
        title = "Teaching Assistant: SAT Math Analyzer API",
        description = "Provide SAT math questions and extract key meta data",
        version="0.1.0",
)

ai_service = MathAIService()

@app.post("/analyze", response_model=MathAnalysisSchema)
async def analyze_single(
    file: UploadFile = File(...),           
):
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename
    
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img_pil = ai_service.read_image(str(temp_file_path))
        result = ai_service.process_single(img_pil, file.filename, SAT_MATH_ANALYZER_PROMPT)

        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])

        return result
    finally:
        if temp_file_path.exists(): temp_file_path.unlink()


@app.post("/analyze/batch")
async def analyze_batch(folder_path: str = Form(...)):
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Invalid folder path")

    try:
        report = ai_service.process_batch(folder_path, SAT_MATH_ANALYZER_PROMPT)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/similar")
async def generate_similar(file: UploadFile = File(...)):
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_file_path = temp_dir / file.filename

    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img_pil = ai_service.read_image(str(temp_file_path))
        generated_text = ai_service.generate_similar_questions(img_pil, SIMILAR_QUESTION_PROMPT)

        return {
            "filename": file.filename,
            "generated_questions": generated_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path.exists(): temp_file_path.unlink()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
