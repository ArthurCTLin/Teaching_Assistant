import gradio as gr
import os
import json
import pandas as pd
from app.services.prompts import SAT_MATH_ANALYZER_PROMPT, SIMILAR_QUESTION_PROMPT
from main import ai_service

def handle_single(image):

    if image is None:
        return "No image uploaded.", None

    result = ai_service.process_single(image, "web_upload.png", SAT_MATH_ANALYZER_PROMPT)
    return json.dumps(result, indent=4), result

def handle_batch(folder_path, progress=gr.Progress()):
    if not os.path.exists(folder_path):
        return "Folder path does not exist.", None, None

    report = ai_service.process_batch(folder_path, SAT_MATH_ANALYZER_PROMPT)

    summary_data = report.get("summary", {})

    print(f"DEBUG - Summary Data: {summary_data}")

    if not summary_data:
        empty_df = pd.DataFrame(columns=["Topic", "Count"])
        return json.dumps(report, indent=4), empty_df, empty_df

    df = pd.DataFrame([{"Topic": str(k), "Count": str(v)} for k, v in summary_data.items()])

    df = df.sort_values(by="Count", ascending=False)

    return json.dumps(report, indent=4), df, df

def handle_generate_similar(image):
    if image is None:
        return "Please upload an image first."
    similar_text = ai_service.generate_similar_questions(image, SIMILAR_QUESTION_PROMPT)
    return similar_text

with gr.Blocks(title="SAT Math AI Assistant") as demo:
    gr.Markdown("# 🎓 SAT Math AI Teaching Assistant")
    gr.Markdown("An automated tagging system for SAT Math questions powered by Gemma-3.")

    with gr.Tabs():
        with gr.TabItem("Single Analysis"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Upload Question Screenshot")
                    btn_single = gr.Button("Analyze Question", variant="primary")
                with gr.Column():
                    output_json = gr.JSON(label="Structured View")
                    output_text = gr.Code(label="Raw JSON Output", language="json")

            btn_single.click(
                fn=handle_single,
                inputs=input_img,
                outputs=[output_text, output_json]
            )

        with gr.TabItem("Batch Processing"):
            gr.Markdown("### Tagging multiple images from a server directory")
            with gr.Row():
                folder_input = gr.Textbox(
                    placeholder="/absolute/path/to/folder",
                    label="Server Folder Path"
                )
                btn_batch = gr.Button("Start Batch Processing", variant="secondary")

            with gr.Row():
                batch_json = gr.Code(label="Batch Report (batch_report.json)", language="json")

                with gr.Column():
                    batch_chart = gr.BarPlot(
                        x="Topic",
                        y="Count",
                        title="Topic Distribution",
                        x_label_angle=-45,
                        tooltip=["Topic", "Count"]
                    )
                    batch_table = gr.DataFrame(label="Summary Table")

            btn_batch.click(
                fn=handle_batch,
                inputs=folder_input,
                outputs=[batch_json, batch_chart, batch_table]
            )
        with gr.TabItem("Generate Similar Questions"):
            gr.Markdown("### Upload a question to generate 3 similar practice problems")
            with gr.Row():
                with gr.Column():
                    input_img_gen = gr.Image(type="pil", label="Example Question")
                    btn_gen = gr.Button("Generate Mirror Questions", variant="primary")
                with gr.Column():
                    output_markdown = gr.Markdown(label="Generated Questions")
            btn_gen.click(
                fn=handle_generate_similar,
                inputs=input_img_gen,
                outputs=output_markdown
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
