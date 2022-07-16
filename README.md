# Whatsapp Chat Analyzer

Have you ever wondered who is the nastiest in the group chat? Which member posts the most? Who takes the longest to reply?

I created this project for learning by doing. Its sole purpose is to write my own parser and to learn the basics of data mining and data analytics. 
I went through several experiments for optimazations, performance increase and parsing issues (god damn you invisible characters!)
Besides, interesting statistics, visualizations and insights can be gained from the analyzed chat messages.

**Feedback is very welcome - _I love to improve myself!_**

## Key Features
- Parsing of whatsapp messages based on exported chat.txt files
- auto detect Android / iOS file format
- Tokenization and lemmatization of messages
- Extract URLs, cursewords and emojis
- Determine message type: questions, replys, initiations, follow-up, media
- a representation of you chat in a Pandas DataFrame for further analysis
- some example analysis functions

## Installation

After pulling use
```conda install -n MYENV --file environment.yml```
to install the environment and its packages (PIP version may coming soon :)). 

If you want to run NLP you need to install Spacey language packages:
- English: ```python -m spacy download en_core_web_md```
- German: ```python -m spacy download de_core_news_md```

## Running

You need to export your chat via Whatsapp. Go to Settings > Chat > export chat.
Transfer the exported .txt file to you device and adapt the path in the jupyter-notebook.
