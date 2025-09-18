"""tool definitions"""

import re
from random import choice

import httpx

from langchain_core.tools import tool
from pydantic import BaseModel
from slugify import slugify

BOOK_API_URL = "https://openlibrary.org/search.json"
QUOTE_AUTHOR_API_URL = "https://api.quotable.io"
QUOTE_BOOK_API_URL = "https://recite.onrender.com/api/v1"


class RandomQuoteInput(BaseModel):
    """No parameters needed."""


class AuthorQuoteInput(BaseModel):
    """Schema for get_author_quote param"""

    author_name: str


class BookQuoteInput(BaseModel):
    """Schema for get_book_quote param"""

    book_name: str


def str_to_query(string: str) -> str:
    """Normalizes string to pass it as a GET parameter"""
    return re.sub(r"\s+", "+", string.lower().strip())


def query_book(book: str) -> tuple[str, str] | None:
    """Queries the open library to look for the book and author
    https://openlibrary.org/dev/docs/api/search
    returns None if book wasn't found
    """
    book_name = str_to_query(book)
    url = f"{BOOK_API_URL}?q={book_name}&fields=title,author_name&limit=1"
    # try except
    response = httpx.get(url)
    response_json = response.json()

    if response_json["num_found"] > 0:
        result = response_json["docs"][0]
        book_title = result["title"]
        author_name = result["author_name"][0] if result["author_name"] else ""
        return (book_title, author_name)

    print(f"{BOOK_API_URL}?q={book_name}&fields=title,author_name&limit=1")


# def get_author_quote(author_name: str) -> dict[str, str]:
@tool(args_schema=AuthorQuoteInput)
def get_author_quote(author_name: str) -> str:
    """Gets random quote from author's name from Quotable API
    https://github.com/lukePeavey/quotable
    """
    author_slug = slugify(author_name)
    # verify false cause ssl certificate is expired
    url = f"{QUOTE_AUTHOR_API_URL}/quotes/random?author={author_slug}"
    # print(url)
    response = httpx.get(url, verify=False)
    response_json = response.json()
    if response_json:
        return f'"{response_json[0]["content"]}" - {author_name}'
    return f"Couldn't find any quote from {author_name}"


# def get_random_quote(_: str = "") -> dict[str, str]:
@tool(args_schema=RandomQuoteInput)
def get_random_quote(_: str = "") -> str:
    """Gets random quote from Quotable API
    https://github.com/lukePeavey/quotable
    """
    # verify false cause ssl certificate is expired
    url = f"{QUOTE_AUTHOR_API_URL}/quotes/random"
    # print(url)
    response = httpx.get(url, verify=False)
    response_json = response.json()
    if response_json:
        return f'"{response_json[0]["content"]}" - {response_json[0]["author"]}'
    return "Couldn't find a random quote."


# def get_book_quote(book_name: str) -> dict[str, str]:
@tool(args_schema=BookQuoteInput)
def get_book_quote(book_name: str) -> str:
    """Gets a quote from the book by hitting recite API
    https://github.com/Sumansourabh14/recite
    """
    book_normalized = str_to_query(book_name)
    url = f"{QUOTE_BOOK_API_URL}/quotes/search?query={book_normalized}"

    response = httpx.get(url)
    response = response.json()

    # add author to response
    if response["success"] and "data" in response:
        book_quotes = [
            (hit["book"], hit["quote"])
            for hit in response["data"]
            if str_to_query(hit["book"]) == book_normalized
        ]
        book, quote = choice(book_quotes)
        return f'{book} - "{quote}"'
    return f"Couldn't find any quote from {book_name}"


def get_random_book_quote(_: str) -> dict[str, str]:
    """Gets a random quote from a book by hitting recite API
    https://github.com/Sumansourabh14/recite
    returns None if there's no response from the server
    """
    url = f"{QUOTE_BOOK_API_URL}/random"

    response = httpx.get(url)
    response = response.json()

    if response:
        return {
            "status": "success",
            "book": response["book"],
            "quote": response["quote"],
            "author": response["author"],
        }
    return {
        "status": "not_found",
        "quote": "",
    }


# TODO: tavily search
