const DEMO_HUMAN_TWEETS = [
  "Just finished a great debugging session. The satisfaction of finding that one missing semicolon! 😅",
  "Beautiful sunset here in San Francisco today. Sometimes you need to step away from the keyboard.",
  "Anyone else excited about the new React 19 features? The compiler looks promising!",
  "Had an amazing ramen for lunch. The small joys of working near Japantown 🍜",
  "Published a new blog post about optimizing database queries. Link in bio!",
];

const DEMO_BOT_TWEETS = [
  "BREAKING: Check out this amazing deal! Click now!!! http://t.co/spam",
  "Follow me for free followers! #followback #follow4follow #teamfollowback",
  "BREAKING: Check out this amazing deal! Click now!!! http://t.co/spam2",
  "You won't believe what happened next! Click here: http://t.co/clickbait",
  "Follow me for free followers! #followback #follow4follow #f4f #autofollow",
];

export default function TweetInput({ tweets, onChange }) {

  const addTweet = () => {
    onChange([...tweets, '']);
  };

  const updateTweet = (index, value) => {
    const updated = [...tweets];
    updated[index] = value;
    onChange(updated);
  };

  const removeTweet = (index) => {
    onChange(tweets.filter((_, i) => i !== index));
  };

  const loadDemoTweets = (type) => {
    onChange(type === 'human' ? [...DEMO_HUMAN_TWEETS] : [...DEMO_BOT_TWEETS]);
  };

  return (
    <div className="tweet-input">
      <div className="section-header">
        <div className="section-icon">💬</div>
        <div>
          <div className="section-title">Tweets</div>
          <div className="section-subtitle">
            Add recent tweets — used for 768-dim LaBSE embedding
          </div>
        </div>
      </div>

      <div className="demo-buttons mb-2">
        <button type="button" className="btn btn-sm btn-secondary" onClick={() => loadDemoTweets('human')}>
          👤 Human Tweets
        </button>
        <button type="button" className="btn btn-sm btn-secondary" onClick={() => loadDemoTweets('bot')}>
          🤖 Bot Tweets
        </button>
      </div>

      {tweets.length === 0 && (
        <div className="tweet-empty glass-card" onClick={addTweet}>
          <span style={{ fontSize: '24px' }}>💬</span>
          <span className="text-muted">No tweets added. Click to add one, or use demo data above.</span>
        </div>
      )}

      <div className="tweet-list">
        {tweets.map((tweet, i) => (
          <div key={i} className="tweet-row">
            <span className="tweet-index">{i + 1}</span>
            <textarea
              className="form-input tweet-textarea"
              value={tweet}
              onChange={e => updateTweet(i, e.target.value)}
              placeholder={`Tweet ${i + 1}...`}
              rows={2}
            />
            <button
              type="button"
              className="btn btn-sm btn-danger tweet-remove"
              onClick={() => removeTweet(i)}
              title="Remove tweet"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      {tweets.length > 0 && (
        <button type="button" className="btn btn-sm btn-secondary mt-2" onClick={addTweet}>
          + Add Tweet
        </button>
      )}

      {tweets.length === 0 && (
        <p className="tweet-warning mt-1">
          ⚠ Without tweets, the model uses only profile features (less accurate).
        </p>
      )}
    </div>
  );
}
