# Quote finder

Simple chatbot that gets quotes from either an author or a book.

Gets quotes from [Quotable](https://github.com/lukePeavey/quotable) and [Recite](https://github.com/Sumansourabh14/recite)

## Install dependencies

Uses [uv](https://github.com/astral-sh/uv) to manage dependencies.

```terminal
uv sync
```

### Direnv

Uses [direnv](https://direnv.net/) to auto activate the python environment

```terminal
direnv allow
```

### Groq API key needed to initialize the llm model

```terminal
cp .env.template .env
```

And add your [Groq](https://groq.com/) api key to the `.env` file.

## Run

```terminal
python main.py
```

```terminal
User: quote from Kafka on the Shore
================================ Human Message =================================

quote from Kafka on the Shore
================================= Reasoning ====================================
Okay, the user is asking for a quote from "Kafka on the Shore." Let me see. First, I need to determine whether to use the get_author_quote or
get_book_quote function. The book title is "Kafka on the Shore," so the correct function would be get_book_quote because it's a specific book. The
author is Haruki Murakami, but since the user mentioned the book title, I should use the book name in the function call.

Wait, the user wrote "Kafka on the Shore" which is the book's title. So I should use get_book_quote with the book_name parameter set to "Kafka on the
Shore." Let me check if that's the correct approach. The get_book_quote function is supposed to get a quote from the book by hitting the recite API.
So yes, using the book name here is appropriate.

I need to make sure that the function is called correctly. The parameters require "book_name" as a string. So the arguments should be {"book_name":
"Kafka on the Shore"}. Let me call that function first. If it doesn't return a quote, then I'll fall back to get_random_quote. But since the user
specifically asked for a quote from that book, I should try the book function first.

================================== Ai Message ==================================
Tool Calls:
  get_book_quote (sqwwz39kn)
 Call ID: sqwwz39kn
  Args:
    book_name: Kafka on the Shore
================================= Tool Message =================================
Name: get_book_quote

Kafka on the Shore - "In traveling, a companion, in life, compassion."
================================= Tool Message =================================
Name: get_book_quote

Kafka on the Shore - "In traveling, a companion, in life, compassion."
User: quote from Walt Disney
================================ Human Message =================================

quote from Walt Disney
================================= Reasoning ====================================
Okay, the user wants a quote from Walt Disney. Let me check the available tools. There's get_author_quote which takes an author's name. Since Walt
Disney is an author, I should use that function. I'll call get_author_quote with author_name set to "Walt Disney". If that doesn't work, I can fall
back to get_random_quote, but I'll try the specific author first.

================================== Ai Message ==================================
Tool Calls:
  get_author_quote (6dwh66ged)
 Call ID: 6dwh66ged
  Args:
    author_name: Walt Disney
================================= Tool Message =================================
Name: get_author_quote

"The more you like yourself, the less you are like anyone else, which makes you unique." - Walt Disney
================================= Tool Message =================================
Name: get_author_quote

"The more you like yourself, the less you are like anyone else, which makes you unique." - Walt Disney
User: surprise me
================================ Human Message =================================

surprise me
================================= Reasoning ====================================
Okay, the user said "surprise me." I need to come up with a quote. Since they want a surprise, maybe a random quote would be best. Let me check the
tools available. There's get_random_quote, which doesn't require any parameters. I should use that. I don't need to call any other tools because the
user didn't specify an author or book. Just call get_random_quote once.

================================== Ai Message ==================================
Tool Calls:
  get_random_quote (6k5v65fkk)
 Call ID: 6k5v65fkk
  Args:
================================= Tool Message =================================
Name: get_random_quote

"I'll prepare and someday my chance will come." - Abraham Lincoln
================================= Tool Message =================================
Name: get_random_quote

"I'll prepare and someday my chance will come." - Abraham Lincoln
```

### Run in streamlit

```terminal
streamlit run ui.py
```

## TODOS

- Multi-agent structure
- Add more tools (wikipedia, tavily)
