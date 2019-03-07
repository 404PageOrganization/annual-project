# coding utf-8


all_characters = set(open('characters.txt', 'r',
                          encoding='utf-8').read().replace('\n', ''))

train_characters = set(open('test.txt', 'r',
                            encoding='utf-8').read().replace('\n', ''))

predict_characters = all_characters - train_characters

f = open('predict.txt', 'w', encoding='utf-8')

for character in predict_characters:
    f.write(character)

f.close
