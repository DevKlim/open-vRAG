
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "OpenGVLab/VideoChat-R1_5"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto",
    attn_implementation="flash_attention_2"
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)

video_path = "your_video.mp4"
question = "your_qa.mp4"
num_percptions = 3

QA_THINK_GLUE = """Answer the question: "[QUESTION]" according to the content of the video. 

Output your think process within the  <think> </think> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. At the same time, in the <glue> </glue> tags, present the precise time period in seconds of the video clips on which you base your answer to this question in the format of [(s1, e1), (s2, e2), ...]. For example: <think>...</think><answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""

QA_THINK = """Answer the question: "[QUESTION]" according to the content of the video.

Output your think process within the  <think> </think> tags.

Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. For example: <think>...</think><answer>A</answer><glue>[(5.2, 10.4)]</glue>.
"""


def inference(video_path, prompt, model, processor, max_new_tokens=2048, device="cuda:0", client = None, pred_glue=None):
    messages = [
        {"role": "user", "content": [
                {"type": "video", 
                "video": video_path,
                'key_time':pred_glue,
                "total_pixels": 128*12 * 28 * 28, 
                "min_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, client = client)
    fps_inputs = video_kwargs['fps']

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)

    generated_ids = [output_ids[i][len(inputs.input_ids[i]):] for i in range(len(output_ids))]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


for percption in range(num_percptions):    
            
    if percption == num_percptions - 1:
        example_prompt = QA_THINK.replace("[QUESTION]", item["problem"]["question"])
    else:
        example_prompt = QA_THINK_GLUE.replace("[QUESTION]", item["problem"]["question"])

    
    ans = inference(video_path, example_prompt, model, processor, device=device, client=client, pred_glue=pred_glue)

    pattern_glue = r'<glue>(.*?)</glue>'
    match_glue = re.search(pattern_glue, ans, re.DOTALL)
    # print(f'ann:{ans}')
    answers.append(ans)
    pred_glue = None
    try:
        if match_glue:
            glue = match_glue.group(1)
            pred_glue = ast.literal_eval(glue)
        
        
    except Exception as e:
        pred_glue = None
print(ans)
