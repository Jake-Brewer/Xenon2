"""Writes a state-tuning set that teaches Xenon to answer Python questions well.

Each answer is short, correct and runnable: one plain sentence, then the code, then one line on
what it prints or why. The chat shows plain text with whitespace kept, so code is not fenced.
Blank lines inside an answer become single newlines, as in RWKV World's own data, because the app
treats "\n\nUser:" as the end of a reply. The evaluation questions are not in the training set.

    python make_python_data.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TRAIN = [
    ("How do I reverse a string in Python?",
     "Slice it with a step of -1:\n"
     "text = \"hello\"\n"
     "print(text[::-1])\n"
     "This prints olleh."),
    ("How do I read a file line by line?",
     "Open it in a with block and loop over the file object:\n"
     "with open(\"notes.txt\", encoding=\"utf-8\") as f:\n"
     "    for line in f:\n"
     "        print(line.rstrip())\n"
     "The with block closes the file for you, even if an error happens."),
    ("What is a list comprehension?",
     "A short way to build a list from a loop:\n"
     "squares = [n * n for n in range(5)]\n"
     "print(squares)\n"
     "This prints [0, 1, 4, 9, 16]. You can add a filter: [n for n in range(10) if n % 2 == 0]."),
    ("How do I count how often each word appears?",
     "Use collections.Counter:\n"
     "from collections import Counter\n"
     "words = \"the cat and the hat\".split()\n"
     "print(Counter(words).most_common(2))\n"
     "This prints [('the', 2), ('cat', 1)]."),
    ("What's the difference between a list and a tuple?",
     "A list can change after you make it; a tuple can't.\n"
     "items = [1, 2]\n"
     "items.append(3)   # fine\n"
     "point = (1, 2)\n"
     "point[0] = 5      # TypeError\n"
     "Use a tuple for a fixed group of values, like coordinates, and a list for a collection that grows."),
    ("How do I handle an error in Python?",
     "Wrap the risky code in try and catch the specific exception:\n"
     "try:\n"
     "    number = int(input(\"Number: \"))\n"
     "except ValueError:\n"
     "    print(\"That wasn't a number.\")\n"
     "Catch the narrowest exception you expect, not a bare except."),
    ("How do I write a function with a default argument?",
     "Give the parameter a value in the def line:\n"
     "def greet(name, greeting=\"Hello\"):\n"
     "    return f\"{greeting}, {name}!\"\n"
     "print(greet(\"Sam\"))\n"
     "print(greet(\"Sam\", \"Hi\"))\n"
     "Never use a mutable default like []; use None and create the list inside."),
    ("Why does my default list keep growing between calls?",
     "The default value is made once, when the function is defined, so every call shares the same list.\n"
     "def add(item, items=None):\n"
     "    if items is None:\n"
     "        items = []\n"
     "    items.append(item)\n"
     "    return items\n"
     "Using None as the default gives each call a fresh list."),
    ("How do I sort a list of dictionaries by a key?",
     "Pass a key function to sorted:\n"
     "people = [{\"name\": \"Ann\", \"age\": 31}, {\"name\": \"Bo\", \"age\": 25}]\n"
     "by_age = sorted(people, key=lambda p: p[\"age\"])\n"
     "print([p[\"name\"] for p in by_age])\n"
     "This prints ['Bo', 'Ann']. Add reverse=True for oldest first."),
    ("How do I loop over a list with the index?",
     "Use enumerate:\n"
     "for i, fruit in enumerate([\"apple\", \"pear\"]):\n"
     "    print(i, fruit)\n"
     "This prints 0 apple, then 1 pear. Pass start=1 to count from one."),
    ("How do I loop over two lists at once?",
     "Use zip:\n"
     "names = [\"Ann\", \"Bo\"]\n"
     "ages = [31, 25]\n"
     "for name, age in zip(names, ages):\n"
     "    print(name, age)\n"
     "zip stops at the shorter list."),
    ("What does if __name__ == \"__main__\" do?",
     "It runs code only when the file is run directly, not when it is imported:\n"
     "def main():\n"
     "    print(\"running\")\n"
     "if __name__ == \"__main__\":\n"
     "    main()\n"
     "Importing this file defines main without calling it."),
    ("How do I make a class in Python?",
     "Use class and an __init__ method to set up each object:\n"
     "class Dog:\n"
     "    def __init__(self, name):\n"
     "        self.name = name\n"
     "    def speak(self):\n"
     "        return f\"{self.name} says woof\"\n"
     "print(Dog(\"Rex\").speak())\n"
     "This prints Rex says woof."),
    ("What is a dataclass?",
     "A class that writes __init__, __repr__ and __eq__ for you from its fields:\n"
     "from dataclasses import dataclass\n"
     "@dataclass\n"
     "class Point:\n"
     "    x: float\n"
     "    y: float\n"
     "print(Point(1, 2))\n"
     "This prints Point(x=1, y=2)."),
    ("How do I read and write JSON?",
     "Use the json module:\n"
     "import json\n"
     "data = {\"name\": \"Xenon\", \"version\": 2}\n"
     "with open(\"data.json\", \"w\", encoding=\"utf-8\") as f:\n"
     "    json.dump(data, f)\n"
     "with open(\"data.json\", encoding=\"utf-8\") as f:\n"
     "    print(json.load(f))\n"
     "json.dumps and json.loads do the same with strings instead of files."),
    ("How do I get the keys and values of a dictionary?",
     "Loop over .items():\n"
     "ages = {\"Ann\": 31, \"Bo\": 25}\n"
     "for name, age in ages.items():\n"
     "    print(name, age)\n"
     "Use ages.get(\"Cy\", 0) to read a key that might be missing without a KeyError."),
    ("How do I remove duplicates from a list?",
     "To keep the original order, use dict.fromkeys:\n"
     "items = [3, 1, 3, 2, 1]\n"
     "print(list(dict.fromkeys(items)))\n"
     "This prints [3, 1, 2]. set(items) also removes duplicates but loses the order."),
    ("What is a generator?",
     "A function that yields values one at a time instead of building a whole list:\n"
     "def countdown(n):\n"
     "    while n > 0:\n"
     "        yield n\n"
     "        n -= 1\n"
     "print(list(countdown(3)))\n"
     "This prints [3, 2, 1]. Generators save memory on large or endless sequences."),
    ("How do I check if a file exists?",
     "Use pathlib:\n"
     "from pathlib import Path\n"
     "if Path(\"notes.txt\").exists():\n"
     "    print(\"found it\")\n"
     "Path also has is_file(), is_dir() and read_text()."),
    ("How do I format a number with two decimal places?",
     "Use an f-string with a format spec:\n"
     "price = 3.14159\n"
     "print(f\"{price:.2f}\")\n"
     "This prints 3.14. Use {price:,.2f} to add thousands separators."),
    ("What's the difference between == and is?",
     "== checks whether two values are equal; is checks whether they are the same object.\n"
     "a = [1, 2]\n"
     "b = [1, 2]\n"
     "print(a == b, a is b)\n"
     "This prints True False. Use is only for None: if x is None."),
    ("How do I split a string into words?",
     "Call split with no arguments to split on any whitespace:\n"
     "print(\"  the quick  fox \".split())\n"
     "This prints ['the', 'quick', 'fox']. Pass a separator to split on something else, like split(\",\")."),
    ("How do I join a list of strings?",
     "Call join on the separator:\n"
     "words = [\"red\", \"green\", \"blue\"]\n"
     "print(\", \".join(words))\n"
     "This prints red, green, blue. Every item must already be a string."),
    ("How do I install a package?",
     "Use pip from a terminal, ideally inside a virtual environment:\n"
     "python -m venv .venv\n"
     "source .venv/bin/activate    # on Windows: .venv\\Scripts\\activate\n"
     "python -m pip install requests\n"
     "python -m pip makes sure the package goes to the Python you are running."),
    ("How do I make an HTTP request?",
     "The requests package is the easiest way:\n"
     "import requests\n"
     "response = requests.get(\"https://example.com\", timeout=10)\n"
     "response.raise_for_status()\n"
     "print(response.text[:100])\n"
     "Always set a timeout so a slow server can't hang your program."),
    ("What is a lambda?",
     "A small unnamed function written in one expression:\n"
     "double = lambda x: x * 2\n"
     "print(double(4))\n"
     "This prints 8. Lambdas are handy as key functions, but a def is clearer for anything longer."),
    ("How do I get the current date and time?",
     "Use datetime:\n"
     "from datetime import datetime\n"
     "now = datetime.now()\n"
     "print(now.strftime(\"%Y-%m-%d %H:%M\"))\n"
     "This prints something like 2026-11-03 14:05."),
    ("How do I take input from the user?",
     "Use input, which always returns a string:\n"
     "name = input(\"What's your name? \")\n"
     "age = int(input(\"How old are you? \"))\n"
     "print(f\"Hi {name}, next year you'll be {age + 1}.\")\n"
     "Convert with int or float when you need a number."),
    ("What does *args and **kwargs mean?",
     "*args collects extra positional arguments into a tuple, **kwargs collects extra keyword arguments into a dict:\n"
     "def show(*args, **kwargs):\n"
     "    print(args, kwargs)\n"
     "show(1, 2, color=\"red\")\n"
     "This prints (1, 2) {'color': 'red'}."),
    ("How do I find the largest number in a list?",
     "Use max:\n"
     "numbers = [4, 9, 2]\n"
     "print(max(numbers))\n"
     "This prints 9. max(people, key=lambda p: p[\"age\"]) finds the item with the largest key."),
    ("How do I check the type of a variable?",
     "Use isinstance, which also accepts subclasses:\n"
     "value = 3.5\n"
     "if isinstance(value, (int, float)):\n"
     "    print(\"it's a number\")\n"
     "type(value) shows the exact type, but isinstance is better for checks."),
    ("How do I copy a list without changing the original?",
     "Make a new list with .copy() or slicing:\n"
     "original = [1, 2, 3]\n"
     "copy = original.copy()\n"
     "copy.append(4)\n"
     "print(original)\n"
     "This prints [1, 2, 3]. For nested lists, use copy.deepcopy."),
    ("How do I write a while loop?",
     "It repeats while the condition is true:\n"
     "count = 3\n"
     "while count > 0:\n"
     "    print(count)\n"
     "    count -= 1\n"
     "This prints 3, 2, 1. Make sure something in the loop changes the condition, or it never ends."),
    ("How do I time how long code takes?",
     "Use time.perf_counter:\n"
     "import time\n"
     "start = time.perf_counter()\n"
     "total = sum(range(1_000_000))\n"
     "print(f\"{time.perf_counter() - start:.3f} s\")\n"
     "For small snippets, python -m timeit gives more reliable numbers."),
    ("How do I replace text in a string?",
     "Use replace, which returns a new string:\n"
     "text = \"I like cats\"\n"
     "print(text.replace(\"cats\", \"dogs\"))\n"
     "This prints I like dogs. Strings never change in place, so keep the result."),
    ("How do I run a shell command from Python?",
     "Use subprocess.run with a list of arguments:\n"
     "import subprocess\n"
     "result = subprocess.run([\"git\", \"status\"], capture_output=True, text=True, check=True)\n"
     "print(result.stdout)\n"
     "A list avoids shell quoting problems; check=True raises an error if the command fails."),
    ("How do I write a unit test?",
     "With pytest, write functions whose names start with test_ and use assert:\n"
     "def add(a, b):\n"
     "    return a + b\n"
     "def test_add():\n"
     "    assert add(2, 3) == 5\n"
     "Save it as test_math.py and run pytest in that folder."),
    ("What is a virtual environment?",
     "A folder with its own Python and packages, so each project's dependencies stay separate.\n"
     "python -m venv .venv\n"
     "Activate it before installing packages, and add .venv to .gitignore."),
    ("How do I convert a string to a number?",
     "Use int or float:\n"
     "print(int(\"42\") + 1)\n"
     "print(float(\"3.5\") * 2)\n"
     "This prints 43 and 7.0. Both raise ValueError if the text isn't a number."),
    ("How do I make a dictionary from two lists?",
     "Zip them and pass the pairs to dict:\n"
     "keys = [\"a\", \"b\"]\n"
     "values = [1, 2]\n"
     "print(dict(zip(keys, values)))\n"
     "This prints {'a': 1, 'b': 2}."),
]

EVAL = [
    "How do I check if a string contains a word?",
    "How do I get the length of a list?",
    "How do I write to a text file?",
    "How do I square every number in a list?",
    "What does the return statement do?",
    "How do I make a list of numbers from 1 to 10?",
    "How do I merge two dictionaries?",
    "How do I check if a list is empty?",
]


def exchange(question: str, answer: str) -> str:
    answer = answer.replace("\n\n", "\n")
    return f"User: {question}\n\nXenon: {answer}\n\n"


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "python_train.jsonl", "w", encoding="utf-8") as handle:
        for question, answer in TRAIN:
            handle.write(json.dumps({"text": exchange(question, answer)}) + "\n")
    with open(out / "python_eval.json", "w", encoding="utf-8") as handle:
        json.dump(EVAL, handle, indent=2)
    print(f"{len(TRAIN)} training exchanges, {len(EVAL)} held-out questions -> {out}")


if __name__ == "__main__":
    main()
