# 📝 Transcribe & Diarize with Whisper + GPT

Этот инструмент позволяет выполнять транскрипцию аудио, разделение по ролям (диаризацию) и интеллектуальную корректировку текста с помощью моделей OpenAI GPT (gpt-3.5-turbo или gpt-4).

## 🚀 Возможности

- Диаризация (распределение ролей "Кандидат" / "Интервьюер") с помощью Resemblyzer. Роли распределяются не всегда верно, возможно надо будет руками поправить
- Транскрипция с Whisper (поддержка русского и английского языков)
- Коррекция ошибок распознавания и постановка вопросов с помощью GPT (тут гпт35 и гпт4, можно выбрать через опции)
- Подсветка вопросов в .docx (фиолетовым), реплик интервьюера (зелёным), для удобства навигации по файлу
- Настройки порога уверенности (`--confidence-threshold`). Это технический параметр, чтобы улучшить качество распознавания сленга.
- Сохранение статистики и логов GPT-запросов. Плюс расчет стоимости на api chatgpt.
- Кэширование результатов Whisper, чтобы не запускать ее каждый раз заново.

## 📦 Установка

```bash
git clone https://github.com/yourname/transcribe-gpt.git
cd transcribe-gpt
pip install -r requirements.txt
```

## 🔑 API ключ

Создайте файл `openai_api_key.txt` и положите рядом с mp3-файлом **или** в `~/.openai_api_key`.

## 📂 Использование

```bash
python3 main.py morgentest.mp3 \
  --lang ru \
  --gpt gpt-3.5-turbo \
  --confidence-threshold 0.92 \
  --skip-whisper 
```
замените файл morgentest.mp3 на любой mp3 менее 30 минут

**Первое выполнение скрипта скачает модель Whisper medium**

## 📄 Использование MP3

*Требование
- Требуется нарезать mp3 на файлы менее 30 мин. 

```bash
ffmpeg -i input.mp3 -f segment -segment_time 1800 -c copy chunk_%03d.mp3
```
Это создаст файлы по 30 минут (1800 секунд).



## 🛠 Параметры

| Параметр                | Описание                                      |
|------------------------|-----------------------------------------------|
| `--lang`               | Язык (`ru` или `en`)                          |
| `--gpt`                | Модель GPT (`gpt-3.5-turbo`, `gpt-4`)         |
| `--confidence-threshold` | Порог вероятности слова (по умолчанию 0.95) |
| `--skip-whisper`       | Пропустить Whisper и использовать кэш (в случае если whisper уже отработал, этот флаг будет переисользовать полученный файл)        |

## 📄 Пример вывода

- `.docx` файл с диалогом и цветовой разметкой
- `.json` с логами правок
- `.txt` со статистикой запросов и оценочной стоимостью api chatgpt
- `morgentest.mp3` пример аудиофайла, на котором можно протестировать

## 📜 Лицензия

MIT
