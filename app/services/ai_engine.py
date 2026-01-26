import torch
import json
import os 
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from typing import Dict, Any, List
from .prompts import SAT_MATH_ANALYZER_PROMPT
from pathlib import Path

class MathAIService:
    def __init__(self):

        self.model_id = "google/gemma-3-4b-it"

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def read_image(self, image_path: str) -> Image.Image:

        img = Image.open(image_path).convert("RGB")
        img = img.resize((640, 640))
        return img

    def ai_analysis(self, img: Image.Image, prompt_text: str) -> str:

        messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a professional math teacher."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img},
                                {"type": "text", "text" : prompt_text}
                                ]
                }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)

    def post_process_output(self, model_output: str, image_path: str) -> Dict[str, Any]:
        try:
            start_idx = model_output.find('{')
            end_idx = model_output.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                return {"error": "No JSON found in output", "raw": model_output}

            json_str = model_output[start_idx:end_idx]
            data = json.loads(json_str)
            
            for field in ["topic", "sub_topic"]:
                if field in data and isinstance(data[field], str):
                    data[field] = data[[field]]

            data["image_path"] = image_path

            return data

        except json.JSONDecodeError as e:
            return {"error": f"JSON decode error: {e}", "raw": model_output}

    def process_single(self, img: Image.Image, image_name: str, prompt_text: str) -> Dict[str, Any]:
        
        raw_output = self.ai_analysis(img, prompt_text)
        return self.post_process_output(raw_output, image_name)

    def process_batch(self, folder_path: str, prompt_text: str) -> Dict[str, Any]:

        folder = Path(folder_path)
        exts = ('.png', '.jpg', '.jpeg', '.webp')
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in exts]

        results = []
        summary = {}

        for img_path in image_files:
            try:
                img_pil = self.read_image(str(img_path))
                res = self.process_single(img_pil, img_path.name, prompt_text)

                if "error" not in res:
                    results.append(res)

                    topics = res.get("topic", ["Unknown"])
                    if isinstance(topics, list):
                        for t in topics:
                            t_str = str(t).strip() # Ensure it's a clean string
                            summary[t_str] = summary.get(t_str, 0) + 1
                    else:
                        t_str = str(topics).strip()
                        summary[t_str] = summary.get(t_str, 0) + 1
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        batch_report = {
                "metadata":{
                    "folder": str(folder_path),
                    "total_files": len(image_files),
                    "successful": len(results)
                },
                "summary": summary,
                "data": results
        }
        with open(folder / "batch_report.json", "w", encoding="utf-8") as f:
            json.dump(batch_report, f, indent=4, ensure_ascii=False)
            
        return batch_report
