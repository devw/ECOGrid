# src/analysis/utils/analysis_constants.py

# Scenari
SCENARIOS = ['NI', 'EI', 'SI']

# Fasce di reddito
income_bracket = lambda x: (
    'Low (0-20K)' if x < 20 else
    'Middle (20-50K)' if x < 50 else
    'High (50-100K)'
)

# Dati teorici per scenario
THEORETICAL_DATA = {
    'NI': {
        'weighted_avg': (14, 20),
        'expected_target': 'High (50-100K)',
        'income_brackets': {
            'Low (0-20K)': (8, 15),
            'Middle (20-50K)': (15, 22),
            'High (50-100K)': (20, 28)
        }
    },
    'EI': {
        'weighted_avg': (29, 40),
        'expected_target': 'Low (0-20K)',
        'income_brackets': {
            'Low (0-20K)': (30, 40),
            'Middle (20-50K)': (28, 38),
            'High (50-100K)': (30, 42)
        }
    },
    'SI': {
        'weighted_avg': (20, 30),
        'expected_target': 'High (50-100K)',
        'income_brackets': {
            'Low (0-20K)': (12, 20),
            'Middle (20-50K)': (18, 28),
            'High (50-100K)': (35, 48)
        }
    }
}

# Classificazione del lift
LIFT_QUALITY = [
    (1.10, 'Poor', '❌'),
    (1.50, 'Moderate', '⚠️'),
    (2.00, 'Good', '✅'),
    (float('inf'), 'Excellent', '⭐')
]
