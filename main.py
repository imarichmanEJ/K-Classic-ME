import os
import json
import uuid
import wave
import random
import asyncio
import traceback
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load environment variables from .env
load_dotenv()

# Initialize Gemini Client (automatically uses GEMINI_API_KEY from env)
client = genai.Client()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global korean_literature_db, prompt1_template, prompt2_template
    
    # Load korean_literature_final.json
    try:
        # test 용 
        with open("korean_literature_final.json", "r", encoding="utf-8") as f:
            korean_literature_db = json.load(f)
    except Exception as e:
        print(f"Error loading korean_literature_final.json: {e}")
        korean_literature_db = []
        
    # Load prompt1.txt
    try:
        with open("prompt1.txt", "r", encoding="utf-8") as f:
            prompt1_template = f.read()
    except Exception as e:
        print(f"Error loading prompt1.txt: {e}")
        prompt1_template = ""

    # Load prompt2.txt
    try:
        with open("prompt2.txt", "r", encoding="utf-8") as f:
            prompt2_template = f.read()
    except Exception as e:
        print(f"Error loading prompt2.txt: {e}")
        prompt2_template = ""
        
    yield

app = FastAPI(title="K-Tale: Become the Hero of Korean Classic Literature", lifespan=lifespan)

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables
korean_literature_db = []
prompt1_template = ""
prompt2_template = ""

# Create necessary directories
os.makedirs("assets", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/audio", exist_ok=True)

# Mount static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Pydantic Schemas
class AnalyzeResponse(BaseModel):
    protagonist_face_description: str
    recommended_ids: List[str]

class GenerateRequest(BaseModel):
    literature_id: str
    protagonist_face_description: str
    narration_language: str = "Korean"

@app.get("/db")
async def get_db():
    return korean_literature_db

def _analyze_sync(prompt: str, file_data: bytes, mime_type: str) -> str:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(data=file_data, mime_type=mime_type),
            prompt
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return response.text

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_face(file: UploadFile = File(...)):
    try:
        # Read file data
        file_data = await file.read()
        mime_type = file.content_type or "image/jpeg"
        
        # Substitute {database_json}
        db_json_str = json.dumps(korean_literature_db, ensure_ascii=False)
        prompt = prompt1_template.replace("{database_json}", db_json_str)
        
        # Run API call in logic thread
        result_text = await asyncio.to_thread(_analyze_sync, prompt, file_data, mime_type)
        
        # Parse JSON output and clean up markdown if any
        try:
            result_json = json.loads(result_text)
        except json.JSONDecodeError:
            clean_text = result_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3]
            result_json = json.loads(clean_text)
            
        return AnalyzeResponse(**result_json)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _generate_story_sync(prompt: str) -> str:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return response.text

async def generate_image(prompt: str, file_path: str, fallback_return: str):
    if os.path.exists(file_path):
        return fallback_return
    
    def _do_image_sync():
        result = client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="16:9"
            )
        )
        for generated_image in result.generated_images:
            with open(file_path, "wb") as f:
                f.write(generated_image.image.image_bytes)
            break
            
    try:
        await asyncio.to_thread(_do_image_sync)
        return fallback_return
    except Exception as e:
        traceback.print_exc()
        print(f"Failed to generate image: {e}")
        return None

async def generate_audio(text: str, file_path: str, fallback_return: str):
    if os.path.exists(file_path):
        return fallback_return
        
    for attempt in range(3):
        try:
            def _do_audio_sync():
                response = client.models.generate_content(
                    model='gemini-2.5-flash-preview-tts',
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            )
                        )
                    )
                )
                return response
                
            response = await asyncio.to_thread(_do_audio_sync)
            
            # Find AUDIO data in candidates
            audio_bytes = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        audio_bytes = part.inline_data.data
                        break
            
            if audio_bytes:
                with wave.open(file_path, "wb") as wav_file:
                    wav_file.setnchannels(1)                # channels=1
                    wav_file.setsampwidth(2)                # sampwidth=2
                    wav_file.setframerate(24000)            # framerate=24000
                    wav_file.writeframes(audio_bytes)
                return fallback_return
            else:
                raise Exception("No AUDIO data in response")
                
        except Exception as e:
            err_str = str(e)
            traceback.print_exc()
            if "500" in err_str or "Internal Server Error" in err_str:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
            print(f"Failed to generate audio (attempt {attempt+1}): {e}")
            break
            
    return None

@app.post("/generate")
async def generate_pipeline(req: GenerateRequest):
    try:
        # Create session ID (YYYYMMDD_HHMMSS_UUID6)
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid_str = str(uuid.uuid4())[:6]
        session_id = f"{dt_str}_{uid_str}"
        
        # Load literature info for prompt context
        target_lit = next((item for item in korean_literature_db if item.get("id") == req.literature_id), None)
        if not target_lit:
            target_lit = {"id": req.literature_id}
            
        lit_json_str = json.dumps(target_lit, ensure_ascii=False)
        
        # Substitute placeholders in prompt2
        prompt = prompt2_template.replace("{literature_json}", lit_json_str)
        prompt = prompt.replace("{protagonist_face_description}", req.protagonist_face_description)
        prompt = prompt.replace("{narration_language}", req.narration_language)
        
        # 1) Generate Story
        result_text = await asyncio.to_thread(_generate_story_sync, prompt)
        
        # Parse output JSON
        try:
            story_data = json.loads(result_text)
        except json.JSONDecodeError:
            clean_text = result_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:-3]
            elif clean_text.startswith("```"):
                clean_text = clean_text[3:-3]
            story_data = json.loads(clean_text)
            
        # Add a random ending question from DB target
        ending_questions = target_lit.get("ending_question", [])
        if ending_questions:
            story_data["ending_question"] = random.choice(ending_questions)
        else:
            story_data["ending_question"] = "이 이야기에 대해 어떻게 생각하시나요?"
            
        # Debugging: Dump raw story output
        with open("story_output.json", "w", encoding="utf-8") as f:
            json.dump(story_data, f, ensure_ascii=False, indent=2)
            
        scenes = story_data.get("scenes", [])
        if not isinstance(scenes, list):
            scenes = [story_data]
            
        # 2) Parallel Task Preparation for Image & Audio Iteration
        async def process_scene(idx, scene):
            scene_no = scene.get("scene_no", idx + 1)
            image_prompt = scene.get("image_prompt", "")
            narration_text = scene.get("narration_text", "")
            
            img_file_path = f"outputs/images/{session_id}_{req.literature_id}_scene{scene_no}.png"
            img_return_path = f"/outputs/images/{session_id}_{req.literature_id}_scene{scene_no}.png"
            
            aud_file_path = f"outputs/audio/{session_id}_{req.literature_id}_scene{scene_no}.wav"
            aud_return_path = f"/outputs/audio/{session_id}_{req.literature_id}_scene{scene_no}.wav"
            
            async def run_img():
                if image_prompt:
                    path = await generate_image(image_prompt, img_file_path, img_return_path)
                    if path:
                        scene["image_path"] = path
                        
            async def run_aud():
                if narration_text:
                    path = await generate_audio(narration_text, aud_file_path, aud_return_path)
                    if path:
                        scene["audio_path"] = path
                        
            # Execute both generations simultaneously for this scene
            await asyncio.gather(run_img(), run_aud())

        # Execute all scenes' async operations concurrently
        all_scene_tasks = [process_scene(idx, scene) for idx, scene in enumerate(scenes)]
        await asyncio.gather(*all_scene_tasks)
        
        return story_data
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# ----------------------------------------------------------------------------
# 실행 안내: uvicorn main:app --reload
# ----------------------------------------------------------------------------
