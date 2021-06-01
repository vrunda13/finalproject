from simpletransformers.question_answering import QuestionAnsweringModel
import json
import os

# Create dummy data to use for training.
train_data = [
    {
        'context': "Skin cancer — the abnormal growth of skin cells — most often develops on skin exposed to the sun. But this common form of cancer can also occur on areas of your skin not ordinarily exposed to sunlight.There are three major types of skin cancer — basal cell carcinoma, squamous cell carcinoma and melanoma.You can reduce your risk of skin cancer by limiting or avoiding exposure to ultraviolet (UV) radiation. Checking your skin for suspicious changes can help detect skin cancer at its earliest stages. Early detection of skin cancer gives you the greatest chance for successful skin cancer treatment.Skin cancer care at Mayo ClinicProducts & Services Book: Mayo Clinic Guide to Stress-Free Living",
        'qas': [
            {
                'id': "00001",
                'is_impossible': False,
                'question': "Which is Skin cancer?",
                'answers': [
                    {
                        'text': "the first",
                        'answer_start': 8
                    }
                ]
            }
        ]
    },
    {
        'context': "Basal cell carcinoma signs and symptoms Basal cell carcinoma usually occurs in sun-exposed areas of your body, such as your neck or face.Basal cell carcinoma may appear as:A pearly or waxy bump A flat, flesh-colored or brown scar-like lesion A bleeding or scabbing sore that heals and returns Squamous cell carcinoma signs and symptoms Most often, squamous cell carcinoma occurs on sun-exposed areas of your body, such as your face, ears and hands. People with darker skin are more likely to develop squamous cell carcinoma on areas that aren't often exposed to the sun.",
        'qas': [
            {
                'id': "00002",
                'is_impossible': False,
                'question': "What are signs of Basal cell carcinoma ?",
                'answers': [
                    {
                        'text': " signs and symptoms",
                        'answer_start': 225
                    }
                ]
            },
            {
                'id': "00003",
                'is_impossible': False,
                'question': "Basal cell carcinoma may appear as?",
                'answers': [
                    {
                        'text': "Basal cell carcinoma may appear as",
                        'answer_start': 167
                    }
                ]
            }
        ]
    }
]
# Save as a JSON file
os.makedirs('data', exist_ok=True)
with open('data/train.json', 'w') as f:
    json.dump(train_data, f)

# Create the QuestionAnsweringModel
model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', args={'reprocess_input_data': True, 'overwrite_output_dir': True})

# Train the model with JSON file
model.train_model('data/train.json')

# The list can also be used directly
# model.train_model(train_data)

# Evaluate the model. (Being lazy and evaluating on the train data itself)
result, text = model.eval_model('data/train.json')

print(result)
print(text)

print('-------------------')

# Making predictions using the model.
to_predict = [{'context': 'This is the context used for demonstrating predictions.', 'qas': [{'question': 'What are signs of Basal cell carcinoma ?', 'id': '0'}]}]

print(model.predict(to_predict))
