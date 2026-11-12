"""Transport-independent house-price model."""
import math

FEATURES = ('area_m2', 'bedrooms', 'age_years', 'distance_to_station_km')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('values must be numbers')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError('values must be finite numbers')
    return value


def validate_model(model):
    if not isinstance(model, dict) or set(model) != {'bias', 'weights'}:
        raise ValueError('model must contain exactly bias and weights')
    if not isinstance(model['weights'], dict) or set(model['weights']) != set(FEATURES):
        raise ValueError('model must contain exactly the supported feature weights')
    number(model['bias'])
    for weight in model['weights'].values():
        number(weight)
    return {'bias': model['bias'], 'weights': dict(model['weights'])}


def predict(features, model):
    if not isinstance(features, dict) or set(features) != set(FEATURES):
        raise ValueError(
            'provide exactly area_m2, bedrooms, age_years, and distance_to_station_km'
        )
    for value in features.values():
        if number(value) < 0:
            raise ValueError('features must be nonnegative')
    prediction = model['bias'] + sum(
        features[name] * model['weights'][name] for name in FEATURES
    )
    number(prediction)
    return {'prediction_baht': round(prediction, 2)}
