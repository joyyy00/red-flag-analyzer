import os
from dotenv import load_dotenv
import praw
import pandas as pd

load_dotenv()  # Load environment variables from .env file

# Initialize Reddit (PRAW)
reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent='joyotano'
)

subreddits = [
    # "dating",
    # "abusiverelationships",
    # "ToxicRelationships",
    # "relationships",
    # "Tinderpickuplines",
    # "tinderstories",
    # "AmIOverreacting",
    # "dating_advice",
    "love"
]

#retrieve content in subreddits
content = []

#retrieve posts
for subreddit_name in subreddits:
    for post in reddit.subreddit(subreddit_name).hot(limit=250):
        text = post.selftext.strip()
        
        #check for mod posts and skip empty text
        if post.distinguished == "moderator" or not text:
            continue

        # automatically label green_flag for these subreddits or TBD if not
        label = "green_flag" if subreddit_name in ["love"] else "TBD"

        content.append({
            'PostID': post.id,
            'Subreddit': subreddit_name,
            'Text': text,
            'Label': label
        })

#dataframe
df = pd.DataFrame(content)
csv_file = 'data/reddit_messages.csv'

#save/append to csv file
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    df = pd.concat([existing_df, df], ignore_index=True)
    #remove duplicates
    df.drop_duplicates(subset=['PostID'], inplace=True)

# Add a clean numeric index ID for readability
if 'ID' not in df.columns:
    df.insert(0, 'ID', range(1, len(df) + 1))
else:
    # Reassign a fresh sequential ID to the combined dataset
    df['ID'] = range(1, len(df) + 1)

df.to_csv(csv_file, index=False)
print(f"Saved {len(df)} total messages to {csv_file}")

