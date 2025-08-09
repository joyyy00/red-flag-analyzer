import praw
import pandas as pd

#create reddit instance
reddit = praw.Reddit(
    client_id = 'WJ0TiuKNOkBA6XWJbSXKAw',
    client_secret = '4R-TFEB5lY1SmKBI2tNaWwQa712XCA',
    user_agent = 'joyotano'    #reddit username
)


### List of subreddits ###
# dating
# abusiverelationships
# ToxicRelationships
# relationships
# Tinderpickuplines
# tinderstories
# relationships
# AmIOverreacting

subreddit_name = 'ToxicRelationships'

#retrieve content in subreddit
content = []
for submission in reddit.subreddit(subreddit_name).hot(limit=5):
    print("Title:", submission.title)
    print("Selftext:", submission.selftext)
    print("-----")
    # text = submission.selftext.strip()
    # if len(text) > 0:
    #     content.append(submission.selftext)

df = pd.DataFrame(content, )
df.to_csv('reddit_messages.csv', index=False)

print("Saved messages to reddit_messages.csv")

