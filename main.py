from transformers import pipeline

# Создаем экземпляр QuestionAnswering из библиотеки transformers
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased')

# Задаем вопрос и предоставляем контекст
with open('text.txt', 'r', encoding='utf8') as file:
    context = file.read()
question = "What is the capital of Russia?"

# Получаем ответ
answer = qa_pipeline({
    'question': question,
    'context': context
})

# Выводим результат
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")