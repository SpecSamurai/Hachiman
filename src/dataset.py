import re

with open('/home/spec/Code/Hachiman/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]

print('Total number of characters:', len(preprocessed))
print(preprocessed[:99])
