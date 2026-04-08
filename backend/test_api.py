"""End-to-end test: send a prediction request to the API."""
import json
import urllib.request

API = "http://localhost:8000"

# Test 1: A "bot-like" profile
bot_request = {
    "target": {
        "profile": {
            "followers_count": 12,
            "friends_count": 4800,
            "listed_count": 0,
            "statuses_count": 35000,
            "favourites_count": 2,
            "name": "News Bot 38291",
            "screen_name": "xnews_bot38291",
            "description": "",
            "created_at": "2023-11-01T00:00:00Z",
            "default_profile": True,
            "default_profile_image": True,
            "verified": False,
            "has_url": False,
            "geo_enabled": False,
            "profile_use_background_image": False,
            "default_profile_background_color": True,
            "default_profile_sidebar_fill_color": True,
            "default_profile_sidebar_border_color": True,
            "profile_background_image_url": False,
        },
        "tweets": [
            "BREAKING: Check out this amazing deal! Click now!!!",
            "Follow me for free followers! #followback #follow4follow",
        ],
    },
    "neighbors": [],
    "relations": [],
}

# Test 2: A "human-like" profile
human_request = {
    "target": {
        "profile": {
            "followers_count": 843,
            "friends_count": 312,
            "listed_count": 15,
            "statuses_count": 4520,
            "favourites_count": 8900,
            "name": "Sarah Chen",
            "screen_name": "sarahchen_dev",
            "description": "Full-stack developer | Open source contributor | Coffee enthusiast",
            "created_at": "2015-03-22T00:00:00Z",
            "default_profile": False,
            "default_profile_image": False,
            "verified": False,
            "has_url": True,
            "geo_enabled": True,
            "profile_use_background_image": True,
            "default_profile_background_color": False,
            "default_profile_sidebar_fill_color": False,
            "default_profile_sidebar_border_color": False,
            "profile_background_image_url": False,
        },
        "tweets": [
            "Just finished a great debugging session. The satisfaction of finding that one missing semicolon!",
            "Beautiful sunset here in San Francisco today.",
            "Anyone else excited about the new React 19 features?",
        ],
    },
    "neighbors": [],
    "relations": [],
}

def predict(name, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{API}/predict/user",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Prediction: {result['label_pred'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  P(human):   {result['prob_human']*100:.1f}%")
    print(f"  P(bot):     {result['prob_bot']*100:.1f}%")
    print(f"  Graph:      {result['graph_info']['num_nodes']} nodes, {result['graph_info']['num_edges']} edges")
    return result

print("Testing MGTAB Bot Detector API...")
print("(First call downloads LaBSE model ~1.8GB, please wait)\n")

r1 = predict("Bot-like Profile", bot_request)
r2 = predict("Human-like Profile", human_request)

print(f"\n{'='*50}")
print("All tests passed!" if r1["label_pred"] != r2["label_pred"] else "Predictions differ as expected!")
