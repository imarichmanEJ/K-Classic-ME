# K-Classic: Become the Hero of Korean Classic Literature

Transform yourself into the protagonist of Korean classical literature through an immersive AI-powered storybook experience.

## Demo


## Features
- Upload a photo to receive 3 personalized Korean classic literature recommendations
- Generate a full illustrated storybook with your face as the protagonist
- Scene-by-scene narration with subtitles and AI voice in 4 languages (Korean, English, Japanese, Chinese)

## Tech Stack
- AI: Gemini 2.5 Flash (multimodal face analysis + story generation)
- Image Generation: Imagen 4.0 (Nano Banana)
- Voice: Gemini TTS
- Backend: FastAPI (Python)
- Frontend: HTML + Vanilla JS

## Getting Started
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Add your `GEMINI_API_KEY` to a `.env` file before running.

## Hackathon
- Built at Gemini 3 Seoul Hackathon (Feb 28, 2026).
- Solo project, developed in 7 hours using Antigravity.
- Minimized wait time through asynchronous parallel generation of images and audio.
