from kiri import Kiri
from docs.emails import emails

"""
Here's example functionality for a customer-service email system. 
This shows two capabilities of Kiri: zero-shot classification, and sentiment detection.

Zero-shot classification is categorizing into a group of labels that were never seen during training.

Sentiment detection... detects text sentiment.
A full list of sentiments is availble in the README -- only a few are noted in this example.
"""

# Common labels for e.g. an e-commerce store's emails
labels = ["Returns", "Promotional", "Technical Issues", "Product Inquiries", "Shipping Questions", "Other"]

# Negative sentiment, give special attention to these.
negative_sentiments = ["annoyance", "disapproval", "disappointment", "anger", "disgust"]

kiri = Kiri()
kiri.classify("This is just to get rid of the example message before printing", ["test"])

# Print example, just to display local results
print("Inbox")
print("==================")
for email in emails:
    classification_results = kiri.classify(email, labels)
    label = max(classification, key=classification_results.get)
    
    emote = kiri.emotion(email)
    high_priority = any([e in emote for e in negative_sentiments])
    
    print(f"Category: {label}")
    if high_priority:
        print("\033[91mPRIORITY\033[0m")
    print(f"{email[:85]}...")
    print("------------------")



    

