# Домашнее задание: Генератор текста на базе Transformer

## Задача

Создайте генератор текста на базе архитектуры Transformer, используя только декодер. Модель должна генерировать ответы авторегрессивно, предсказывая следующее слово на основе контекста.

## Требования

### 1. Архитектура модели

Создайте класс `GeneratorTransformer`, который авторегрессивно генерирует продолжение текста. Обучите его на книгах или каких-нибудь текстах, которые вы найдете в интернете

Был реализован класс `GeneratorTransformer`(в файле generator.py), включающий:
- Входную embedding-матрицу
- Позиционные эмбеддинги
- Стек декодерных слоёв
- Линейный слой на выходе с softmax

### 2. Токенизация

Использован **готовый токенизатор** из файла:

```python
tokenizer = Tokenizer.from_file("transformer_basics/mistral_tokenizer.json")
tokenizer.add_special_tokens(["<s>", "</s>", "<pad>"])
```

Также реализовано добавление `bos`, `eos`, `pad` токенов в датасете.


## 4. Обучение

Обучение проводилось на RTX 4060 с использованием `torch.amp` (FP16), `GradScaler`, `autocast`.

**Гиперпараметры:**
- `batch_size = 32` увеличил для ускорения
- `max_length = 128`
- `learning_rate = 1e-4`
- `num_epochs = 7` тестировал для beam_search(так и не получилось)

### Dataset:
Использован `wikitext-2-raw-v1` из `datasets`:

Создан свой `TextDataset`, в котором:
- Весь текст объединяется в одну строку
- Токенизируется
- Режется на последовательности длиной `max_length`
- Каждой последовательности добавляются токены `<s>` в начало и `</s>` в конец
- Возвращаются пары `input_ids`, `target_ids` со сдвигом на один

```
Epoch 1: 100%|██████████| 687/687 [00:34<00:00, 19.63it/s, loss=6.87]
Epoch 1 finished. Loss: 6.8463
Generated: In the future of symbolvert , in providedative wifeorous for the minor . Later and conditions wasv of theyley spread parking — tro announced . highlighted , within calledott , and yield are the international Vill to one named ofatch important becomes Bon writing history Som
Epoch 2: 100%|██████████| 687/687 [00:34<00:00, 19.86it/s, loss=6.01]
Epoch 2 finished. Loss: 5.9890
Generated: In the future concept on 28484 Society , spiniviaedistics , an most Kil Ver , in 500reation to 1 , 1 , although in the originalfl , Dylan . 
 The priorities = =
Epoch 3: 100%|██████████| 687/687 [00:34<00:00, 19.88it/s, loss=5.73]
Epoch 3 finished. Loss: 5.7130
Generated: In the future most capture . When Beco is approachedmentanistent due to bemate in an honorberrygeon . Compet for amounts as a nightiba and fewnaments 's own Sweet of Nelson blest Museum as a species of the shle of them vocals
Epoch 4: 100%|██████████| 687/687 [00:34<00:00, 19.99it/s, loss=5.51]
Epoch 4 finished. Loss: 5.5150
Generated: In the future records into the left of the maximum to anboard and Fame . Reaga of the New Zealand such as the company of the Qufully , of this period ( tenons ) , once designed , Christathens SAels a troops a spiritualau Pan
Epoch 5: 100%|██████████| 687/687 [00:34<00:00, 19.99it/s, loss=5.36]
Epoch 5 finished. Loss: 5.3634
Generated: In the future of the valley . Its ends because siming had been made being think in the Paris and him to a opening . However , " He translate on August , " Ithe interests rayly lular 's evident worked of most involved the cap , but
Epoch 6: 100%|██████████| 687/687 [00:34<00:00, 19.98it/s, loss=5.25]
Epoch 6 finished. Loss: 5.2395
Generated: In the future of the Detroit after by fish to its predecessial crAL , in an 159 . The following order reached the Spanish to the alminated Bay of the roadensive , the bottom . 
 Beyon and following episode director
Epoch 7: 100%|██████████| 687/687 [00:34<00:00, 19.88it/s, loss=5.13]
Epoch 7 finished. Loss: 5.1347
Generated: In the future of its words , a heart decided in 2ndspian Tower the subsequentong , the 3 million years of 1971 Medal in the Oriior School . After a result round that season to Twain later became a General arg
Training complete
```

## 5. Авторегрессивная генерация

Реализован метод `generate(...)`, который:
- Токенизирует ввод
- Сдвигает окно контекста на 1 токен вправо
- Генерирует по одному токену за раз
- Останавливается при достижении `eos` или max длины

## 6. Сдвиг контекста при авторегрессии

Контекст сдвигается на один токен влево при каждой итерации генерации.

## 7. Тестирование

Создал простой интерфейс чата в файле chat.py
#### Пример
```
Режим генерации: Stock
Вы: In the  
Бот: In the site of the U.S.C. 23 – 25 million , which was estimated to beygons , but were caught in which would only , and a 1971 . In the major year , the 1780s , the Ober of the first dispatched leaders the New York , which was a kick ratio of the Push , and the Civil War . The route adopted was a generation of the bulk ( 1989 ) , 
Вы: 
```
