import os
import json
import time
from tqdm import tqdm
from groq import Groq
import sys
import subprocess

# Configuration
MODEL_NAME = "openai/gpt-oss-120b"
DATA_PATH = "../data/dev.json"
OUTPUT_DIR = "../llm-ollama/zero-shot/openai-gpt-oss-120b"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Combined prompt as per Groq reasoning recommendations (avoid system prompts)
FULL_PROMPT_TEMPLATE = (
    """
    You are an expert NLU annotator. Your job is to rate how plausible a candidate meaning (sense)
    is for the HOMONYM used in the target sentence within the short story.

    Return ONLY a single JSON object with one key: "score" and an integer value 1, 2, 3, 4 or 5.
    Integer mapping:
      1 = Definitely not
      2 = Probably not
      3 = Ambiguous / Unsure
      4 = Probably yes
      5 = Definitely yes

    The response must be a JSON object and nothing else, for example: {{"score": 4}}

    [STORY]
    {full_story_text}

    [HOMONYM]
    {homonym}

    [CANDIDATE SENSE]
    {sense_text}

    [TASK]
    Based on the STORY above, decide how plausible it is that the HOMONYM is used with the
    CANDIDATE SENSE in the target sentence.

    Return ONLY a single JSON object with one key "score" and an integer value (1-5)
    as described by the system message. Example output: {{"score": 3}}
    """
)

def create_full_story_text(item):
    fullstory = f"{item.get('precontext', '')} {item.get('sentence', '')} {item.get('ending', '')}"
    return fullstory.strip()

def create_message(item):
    sense = f"{item.get('judged_meaning', '')} as in \"{item.get('example_sentence', '')}\"".strip()
    homonym = item.get("homonym", "")
    full_story_text = create_full_story_text(item)
    
    return FULL_PROMPT_TEMPLATE.format(
        full_story_text=full_story_text,
        homonym=homonym,
        sense_text=sense
    )

# Load data
print(f"Loading data from {DATA_PATH}...")
with open(DATA_PATH, "r") as f:
    data = json.load(f)

# Prepare output files
pred_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
ref_file = os.path.join(OUTPUT_DIR, "ref.jsonl")
failed_file = os.path.join(OUTPUT_DIR, "failed_ids.jsonl")

# Check if we can resume
processed_ids = set()
if os.path.exists(pred_file):
    with open(pred_file, "r") as f:
        for line in f:
            try:
                processed_ids.add(json.loads(line)["id"])
            except:
                pass
    print(f"Resuming from {len(processed_ids)} processed items.")

# Open files for appending
with open(pred_file, "a") as f_pred, open(ref_file, "a") as f_ref, open(failed_file, "a") as f_fail:
    
    # Iterate over data
    for item_id, item in tqdm(data.items()):
        if item_id in processed_ids:
            continue
            
        user_content = create_message(item)
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                model=MODEL_NAME,
                temperature=0, # Deterministic
                reasoning_effort="low", # Minimize token usage for rate limits
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "homonym_score",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {
                                    "type": "integer",
                                    "enum": [1, 2, 3, 4, 5]
                                }
                            },
                            "required": ["score"],
                            "additionalProperties": False
                        }
                    }
                },
            )
            
            response_content = chat_completion.choices[0].message.content
            
            # Parse JSON
            try:
                response_json = json.loads(response_content)
                score = response_json.get("score")
                
                if score is not None:
                    # Save prediction
                    f_pred.write(json.dumps({"id": item_id, "prediction": score}) + "\n")
                    f_pred.flush()
                    
                    # Save reference (gold)
                    f_ref.write(json.dumps({"id": item_id, "label": item["choices"]}) + "\n")
                    f_ref.flush()
                else:
                    raise ValueError("No score found in response")
                    
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for {item_id}: {response_content}")
                f_fail.write(json.dumps({"id": item_id, "error": "json_parse", "content": response_content}) + "\n")
                f_fail.flush()
                
        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            f_fail.write(json.dumps({"id": item_id, "error": str(e)}) + "\n")
            f_fail.flush()
            # Sleep a bit on error to avoid hammering if it's a rate limit
            time.sleep(2)

print("Done!")

# Run scoring script
score_file = os.path.join(OUTPUT_DIR, "score.json")
scoring_script = "../score/scoring.py"

print(f"Running scoring script: {scoring_script}")
subprocess.run([sys.executable, scoring_script, ref_file, pred_file, score_file])
