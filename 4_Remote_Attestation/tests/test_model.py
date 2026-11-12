import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'server'))

from model import predict, validate_model

MODEL = {
    'bias': 500000,
    'weights': {
        'area_m2': 40000,
        'bedrooms': 200000,
        'age_years': -15000,
        'distance_to_station_km': -100000,
    },
}


class ModelTests(unittest.TestCase):
    def test_prediction_at_default_distance(self):
        features = {
            'area_m2': 120, 'bedrooms': 3, 'age_years': 10,
            'distance_to_station_km': 2.0,
        }
        self.assertEqual(predict(features, validate_model(MODEL)),
                         {'prediction_baht': 5550000.0})

    def test_prediction_at_five_km(self):
        features = {
            'area_m2': 120, 'bedrooms': 3, 'age_years': 10,
            'distance_to_station_km': 5.0,
        }
        self.assertEqual(predict(features, validate_model(MODEL)),
                         {'prediction_baht': 5250000.0})

    def test_rejects_missing_distance(self):
        features = {'area_m2': 120, 'bedrooms': 3, 'age_years': 10}
        with self.assertRaises(ValueError):
            predict(features, validate_model(MODEL))

    def test_rejects_negative_distance(self):
        features = {
            'area_m2': 120, 'bedrooms': 3, 'age_years': 10,
            'distance_to_station_km': -1,
        }
        with self.assertRaises(ValueError):
            predict(features, validate_model(MODEL))


if __name__ == '__main__':
    unittest.main()
