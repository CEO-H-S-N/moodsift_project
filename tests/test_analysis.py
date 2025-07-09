import unittest
from unittest.mock import patch, MagicMock
import torch
from services.analysis import SentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    @patch('transformers.RobertaForSequenceClassification.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model):
        # Mock model and tokenizer
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        mock_model.return_value = self.mock_model
        mock_tokenizer.return_value = self.mock_tokenizer
        
        # Mock pipeline
        self.mock_pipeline = MagicMock()
        self.mock_pipeline.return_value = [{'label': 'positive', 'score': 0.95}]
        
        # Initialize analyzer with mocks
        with patch('transformers.pipeline', return_value=self.mock_pipeline):
            self.analyzer = SentimentAnalyzer()

    def test_analyze_sentiment(self):
        test_text = "I love this product!"
        result = self.analyzer.analyze_sentiment(test_text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label'], 'positive')

    def test_get_top_sentiment(self):
        test_text = "This is terrible!"
        self.mock_pipeline.return_value = [{'label': 'negative', 'score': 0.90}]
        result = self.analyzer.get_top_sentiment(test_text)
        self.assertEqual(result['label'], 'negative')
        self.assertGreaterEqual(result['score'], 0.5)

    @patch('services.analysis.SentimentAnalyzer.get_top_sentiment')
    def test_analyze_batch(self, mock_top_sentiment):
        mock_top_sentiment.return_value = {'label': 'neutral', 'score': 0.8}
        texts = ["Text 1", "Text 2"]
        results = self.analyzer.analyze_batch(texts)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['label'], 'neutral')

    def test_get_sentiment_distribution(self):
        texts = ["Good", "Bad", "Good"]
        with patch('services.analysis.SentimentAnalyzer.get_top_sentiment') as mock_top:
            mock_top.side_effect = [
                {'label': 'positive'},
                {'label': 'negative'},
                {'label': 'positive'}
            ]
            dist = self.analyzer.get_sentiment_distribution(texts)
            self.assertEqual(dist['positive'], 2)
            self.assertEqual(dist['negative'], 1)

if __name__ == '__main__':
    unittest.main()