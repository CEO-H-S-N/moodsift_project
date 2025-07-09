import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipelines.data_collection import DataCollector
from pipelines.preprocessing import TextPreprocessor

class TestDataCollection(unittest.TestCase):
    @patch('praw.Reddit')
    @patch('tweepy.Client')
    def setUp(self, mock_twitter, mock_reddit):
        self.mock_reddit = MagicMock()
        self.mock_twitter = MagicMock()
        mock_reddit.return_value = self.mock_reddit
        mock_twitter.return_value = self.mock_twitter
        
        # Mock Reddit submission
        mock_submission = MagicMock()
        mock_submission.id = '123'
        mock_submission.title = 'Test Title'
        mock_submission.selftext = 'Test content'
        mock_submission.created_utc = 1234567890
        mock_submission.score = 42
        mock_submission.num_comments = 7
        self.mock_reddit.subreddit.return_value.hot.return_value = [mock_submission]
        
        # Mock Twitter response
        mock_tweet = MagicMock()
        mock_tweet.id = '456'
        mock_tweet.text = 'Test tweet'
        mock_tweet.created_at = '2023-01-01'
        mock_tweet.public_metrics = {'like_count': 10, 'retweet_count': 2}
        self.mock_twitter.search_recent_tweets.return_value.data = [mock_tweet]
        
        self.collector = DataCollector()

    def test_collect_reddit_posts(self):
        result = self.collector.collect_reddit_posts(['test'], limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['source'], 'reddit')

    def test_collect_twitter_posts(self):
        result = self.collector.collect_twitter_posts('test', max_results=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['source'], 'twitter')

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.test_df = pd.DataFrame({
            'text': [
                "Check this out: https://example.com",
                "Hello @user! #happy",
                "Short"
            ]
        })

    def test_clean_text(self):
        cleaned = self.preprocessor.clean_text("Hello @user! Visit https://site.com #tag")
        self.assertNotIn('@user', cleaned)
        self.assertNotIn('https://', cleaned)
        self.assertNotIn('#tag', cleaned)

    def test_preprocess_data(self):
        result = self.preprocessor.preprocess_data(self.test_df)
        self.assertEqual(len(result), 2)  # Should remove short text
        self.assertTrue(all('http' not in text for text in result['cleaned_text']))

    @patch('transformers.AutoTokenizer')
    def test_tokenize_data(self, mock_tokenizer):
        mock_tokenizer.return_value = MagicMock(return_value={'input_ids': [[1,2,3]]})
        texts = pd.Series(['test text'])
        result = self.preprocessor.tokenize_data(texts)
        self.assertIn('input_ids', result)

if __name__ == '__main__':
    unittest.main()