from kiri import Kiri, Document, InMemoryDocStore

"""
Here's example functionality for a customer-service email system. 
This shows two capabilities of Kiri: zero-shot classification, and sentiment detection.

Zero-shot classification is categorizing into a group of labels that were never seen during training.

Sentiment detection... detects text sentiment.
A full list of sentiments is availble in the README -- only a few are noted in this example.
"""

# Common labels for e.g. an e-commerce store's emails
labels = ["Returns", "Promotional", "Technical Issues", "Product Inquiries", "Shipping Questions"]

# Negative sentiment, give special attention to these.
negative_sentiments = ["annoyance", "disapproval", "disappointment", "anger", "disgust"]

kiri = Kiri()

emails = ["Hi, how long will it take for my order to arrive? I've been waiting four days.", 
          "I got my order a few days ago. Unfortunately, the materials don't seem to be as high quality as I expected. The colors are also duller than the photos. Can I get my money back?",
          "Where does your store source materials from? Can you do made-to-order designs?",
          "How could you sell something this broken? This is the worst thing I've ever seen, and you should be ashamed to make money off it.",
          "Hello. I'm a copywriter that specializes in marketing small e-commerce stores. I've taken a look at your website, and think we could work together if you're interested. Feel free to write me back.",
          "I'm trying to order a few items for an upcoming birthday. For some reason, I can't checkout. I can load the cart just fine, but the checkout button clears everything I've added, and I can't continue the purchase."]


# Print example, just to display local results
print("Inbox")
print("==================")
for email in emails:
    classification = kiri.classify(email, labels)
    label = max(classification, key=classification.get)
    emote = kiri.emotion(email)
    high_priority = any([f in emote for f in negative_sentiments])
    print(f"Category: {label}")
    if high_priority:
        print("PRIORITY")
    print(f"{email[:85]}...")
    print("------------------")



    

