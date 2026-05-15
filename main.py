from src.inference import predict_text

text = input("Enter your text : ")

result = predict_text(text)
print(result)