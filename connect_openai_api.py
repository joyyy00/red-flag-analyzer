import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import time

load_dotenv()  # Load environment variables from .env file

# Initialize OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

###############################
# add a check for labeled csv file to see if TBD/unlabeled exists. This script only checks the 
# reddit_messages.csv which obviously has only TBD (besides auto labeled green_flags)
###############################

#CSV file
df = pd.read_csv("reddit_messages.csv")

categories=["red_flag", "green_flag"]

def label_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                #assistant instruction
                {"role": "system", "content": "You are an assistant that classifies relationship messages and stories as red flags or green flags (neutral or positive)."},
                #prompt
                {"role": "user", "content": f"""Classify the following text into one of these two categories: {categories}. Text: \"{text}\" Respond ONLY with 'red_flag' or 'green_flag'."""}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # error message + unlabeled if something goes wrong
        print(f"Error labeling text: {e}")
        return "unlabeled"

new_rows = df[df["Label"].isin(["TBD", "unlabeled"])].copy()
print(f"Found {len(new_rows)} unlabeled rows to process.")

# Only label rows that are still TBD
for idx, row in new_rows.iterrows():
    label = label_text(row["Text"])
    df.at[idx, "Label"] = label
    print(f"Labeled row {row['ID']} as {label}")
    time.sleep(1)  # avoid hitting API rate limits
  
# Save updated CSV
output_file = "reddit_messages_labeled.csv"
df.to_csv(output_file, index=False)
print(f"Labeled data saved to {output_file}")
