import requests

API_URL = 'http://localhost:5000/predict'

# Тестовые юридические тексты
test_texts = [
    "The court cited the precedent case of Brown v. Board of Education",
    "The judge applied the new statute to the current situation",
    "This decision was followed by several lower courts",
    "The case was referred to the Supreme Court for final review",
    "The matters discussed during the hearing were confidential"
]

for text in test_texts:
    response = requests.post(API_URL, json={'text': text})
    result = response.json()
    
    print(f"Текст: {result['text']}")
    print(f"Категория: {result['case_outcome']} (уверенность: {result['confidence']:.2f})")
    print("Распределение вероятностей:")
    
    # Сортируем вероятности по убыванию для удобства чтения
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    for k, v in sorted_probs:
        print(f"  {k}: {v:.4f}")
    print("-" * 50)