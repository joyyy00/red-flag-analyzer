import praw
import pandas as pd
import os

#create reddit instance
reddit = praw.Reddit(
    client_id = 'WJ0TiuKNOkBA6XWJbSXKAw',
    client_secret = '4R-TFEB5lY1SmKBI2tNaWwQa712XCA',
    user_agent = 'joyotano'    #reddit username
)

subreddits = [
    "dating",
    "abusiverelationships",
    "ToxicRelationships",
    "relationships",
    "Tinderpickuplines",
    "tinderstories",
    "AmIOverreacting",
    "dating_advice"
]

#retrieve content in subreddits
content = []

#retrieve posts
for subreddit_name in subreddits:
    for post in reddit.subreddit(subreddit_name).hot(limit=15):
        text = post.selftext.strip()
        
        #check for mod posts and skip empty text
        if post.distinguished == "moderator" or not text:
            continue

        content.append({
            'PostID': post.id,
            'Subreddit': subreddit_name,
            'Text': text,
            'Label': 'TBD'
        })

#dataframe
df = pd.DataFrame(content)
csv_file = 'reddit_messages.csv'

#save/append to csv file
if os.path.exists(csv_file):
    existing_df = pd.read_csv(csv_file)
    df = pd.concat([existing_df, df], ignore_index=True)
    #remove duplicates
    df.drop_duplicates(subset=['PostID'], inplace=True)

# Add a clean numeric index ID for readability
df.insert(0, 'ID', range(1, len(df) + 1))

df.to_csv(csv_file, index=False)
print(f"Saved {len(df)} total messages to {csv_file}")

