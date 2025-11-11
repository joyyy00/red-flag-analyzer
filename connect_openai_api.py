from openai import OpenAI
import pandas as pd
import os
import time

client = OpenAI(
  api_key="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"
)

#CSV file
df = pd.read_csv("reddit_messages.csv")

categories=["red_flag", "green_flag"]

def label_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                #assistant instruction
                {"role": "system", "content": "You are an assistant that classifies relationship messages and stories as red flags or neutral."},
                #prompt
                {"role": "user", "content": f"""Classify the following text into one of these two categories: {categories}. Text: \"{text}\" Respond ONLY with 'red_flag' or 'green_flag'."""}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        # error message + unlabeled if something goes wrong
        print(f"Error labeling text: {e}")
        return "unlabeled"

# Only label rows that are still TBD
for idx, row in df.iterrows():
    if row["Label"] == "TBD":
        label = label_text(row["Text"])
        df.at[idx, "Label"] = label
        print(f"Labeled row {row['ID']} as {label}")
        time.sleep(1)  # avoid hitting API rate limits
  
# Save updated CSV
df.to_csv("reddit_messages_labeled.csv", index=False)
print("Labeled data saved to reddit_messages_labeled.csv")
