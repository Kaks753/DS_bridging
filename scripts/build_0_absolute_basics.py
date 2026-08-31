"""Builder for Module 0: Absolute Basics — Python from Zero.

This module assumes ZERO prior programming knowledge. It is the on-ramp that
sits *before* NumPy/Pandas. Written entirely in the 4-layer teaching pattern.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from nbtools import Notebook

nb = Notebook()

nb.md(r"""
# Module 0 — Absolute Basics: Python from Zero

**Read this even if you think you know Python.** We start from *nothing* — what a
program even is — and build up slowly to the point where the rest of the bootcamp
makes sense. No step is skipped. If you already know a bit, skim the 🎓 boxes and
do the ✍️ practice prompts to check yourself.

By the end you'll comfortably understand:
- what code, a notebook, and a "cell" actually are
- **variables** and the main **data types** (text, numbers, True/False)
- **lists** and **dictionaries** (how Python holds many things)
- **if** decisions and **for** loops
- **functions** (reusable mini-machines)
- **importing libraries** (borrowing other people's code)
- how to **read an error message** without panicking

> Teaching style for the whole bootcamp: 🌱 plain-English first → 🔤 code explained
> line by line → 🎓 the deeper why → ✅ a one-line takeaway. Take your time.
""")

# ---------------------------------------------------------------------------
nb.md("## 0.1 What is a program? What is this notebook?")

nb.plain(r"""
A **program** is just a list of instructions you give a computer, written in a
language it understands. We use the **Python** language because it reads almost
like English and is the #1 language in data science.

You're reading a **Jupyter notebook** — a document made of stacked boxes called
**cells**. Two kinds:
- *text cells* (like this one) — notes for humans.
- *code cells* (grey, below) — instructions Python runs when you press
  **Shift+Enter**. Any result appears right underneath.

Think of a notebook like a **kitchen worktop**: each code cell is one step of a
recipe. You run steps top-to-bottom, and ingredients you prepared earlier are
still there for later steps.
""")

nb.code(r"""
# This grey box is a CODE cell. Lines starting with # are 'comments' —
# notes for humans that Python ignores. The line below is a real instruction.
print("Hello, Stephen! This line ran on the computer.")
""")

nb.readcode(r"""
- `# ...` → a **comment**. Python skips it. Use comments to explain your thinking.
- `print(...)` → a built-in command that **shows** whatever is inside the brackets.
- `"Hello, Stephen! ..."` → the text to show. Text wrapped in quotes is called a
  **string** (more on that next).

Press **Shift+Enter** on a code cell to run it. The line under the cell is the
**output** — what the code produced.
""")

nb.jargon("cell", "one box in a notebook; either human text or runnable code")
nb.jargon("print()", "a command that displays a value on the screen")
nb.takeaway("A notebook is a recipe of cells; code cells run top-to-bottom and show their results underneath.")

# ---------------------------------------------------------------------------
nb.md("## 0.2 Variables — labelled boxes that store values")

nb.plain(r"""
A **variable** is a **labelled box** where you store a value so you can use it
again later by its name. You *create* a variable with `=`, which means
"put the thing on the right into the box named on the left" (it does **not** mean
"equals" like in maths).

Example: `age = 27` means *"make a box called `age` and put `27` in it."* From then
on, wherever you write `age`, Python swaps in `27`.
""")

nb.code(r"""
name = "Stephen"     # a box called name, holding the text "Stephen"
age = 27             # a box called age, holding the number 27
height_m = 1.78      # a box called height_m, holding a decimal number

print(name)          # show what's in the name box
print(age)
print(height_m)

age = 28             # we can OVERWRITE a box; now age holds 28
print("updated age:", age)
""")

nb.readcode(r"""
- `name = "Stephen"` → create box `name`, store the string `"Stephen"`.
- `age = 27` → create box `age`, store the number `27`.
- `print(name)` → look inside box `name` and show it → `Stephen`.
- `age = 28` → replace what's in `age`; the old `27` is gone.
- `print("updated age:", age)` → `print` can show several things separated by
  commas; it prints them on one line with a space between.
""")

nb.warn("`=` is **assignment** (put value in box), not equality. To *test* if two "
        "things are equal you use `==` (double equals) — we'll see that in 0.5.")
nb.jargon("variable", "a named box that stores a value you can reuse")
nb.takeaway("Create a variable with `name = value`; use the name later to get the value back.")
nb.try_this("Make a variable `city = \"Nairobi\"` and print a sentence like "
            "`Stephen lives in Nairobi` using print with commas.")

# ---------------------------------------------------------------------------
nb.md("## 0.3 Data types — the main *kinds* of value")

nb.plain(r"""
Every value has a **type** — what *kind* of thing it is. The four you'll use most:

- **string** (`str`): text, always in quotes → `"hello"`, `"Nairobi"`.
- **integer** (`int`): a whole number → `27`, `-4`, `1000`.
- **float** (`float`): a number with a decimal point → `1.78`, `3.0`.
- **boolean** (`bool`): a yes/no value → only `True` or `False`.

Why care? Because Python treats them differently. `"5" + "5"` glues text into
`"55"`, but `5 + 5` adds numbers into `10`. Same-looking, totally different!
""")

nb.code(r"""
a_string  = "data science"
an_int    = 42
a_float   = 3.14
a_bool    = True

# type() tells you the kind of a value:
print(type(a_string))
print(type(an_int))
print(type(a_float))
print(type(a_bool))

# The classic gotcha — text vs numbers:
print("5" + "5")   # strings get GLUED together
print(5 + 5)       # numbers get ADDED
""")

nb.readcode(r"""
- `type(x)` → asks Python "what kind of value is `x`?" and prints it (e.g.
  `<class 'str'>` means it's a string).
- `"5" + "5"` → both are **strings**, so `+` means *join text* → `"55"`.
- `5 + 5` → both are **numbers**, so `+` means *add* → `10`.

That one difference causes a huge share of beginner bugs, so always know your types.
""")

nb.deeper(r"""
You can **convert** between types when it's sensible:
`int("5")` → the number `5`; `str(5)` → the string `"5"`; `float("3.14")` → `3.14`.
This is called **casting**. It matters constantly in data work: a CSV loads
everything as text, so a column of ages arrives as strings like `"27"` and you must
cast it to numbers before you can average it. (Module on cleaning covers this.)
""")

nb.jargon("data type", "the kind of a value: text (str), whole number (int), decimal (float), or True/False (bool)")
nb.takeaway("Know each value's type — text and numbers look alike but behave completely differently.")

# ---------------------------------------------------------------------------
nb.md("## 0.4 Lists and dictionaries — holding *many* things")

nb.plain(r"""
So far one box held one value. Often you have **many** values. Two everyday
containers:

- A **list** is an **ordered shopping list** — items in a row, each with a position
  number (its *index*). Written with square brackets: `[ ]`.
  ⚠️ Python counts positions from **0**, not 1. So the first item is index `0`.
- A **dictionary** is a **labelled drawer set** — each value is stored under a
  *name* (a "key") instead of a position. Written with curly braces: `{ }`.
  Great when "the 2nd thing" is less meaningful than "the *income* thing".
""")

nb.code(r"""
# A LIST — ordered, accessed by position (starting at 0)
cities = ["Nairobi", "Mombasa", "Kisumu"]
print("whole list :", cities)
print("first item :", cities[0])     # index 0 = first
print("second item:", cities[1])
print("how many   :", len(cities))   # len() counts the items

cities.append("Nakuru")              # add a new item to the end
print("after adding:", cities)
""")

nb.code(r"""
# A DICTIONARY — accessed by name (key), not position
customer = {
    "name": "Stephen",
    "age": 27,
    "city": "Nairobi",
}
print("the whole dict:", customer)
print("just the city :", customer["city"])   # look up by KEY, not number
customer["plan"] = "Premium"                  # add a new key:value pair
print("after adding plan:", customer)
""")

nb.readcode(r"""
List cell:
- `cities = [...]` → make a list of three strings.
- `cities[0]` → the item at position 0 (the **first**) → `"Nairobi"`.
- `len(cities)` → **len**gth = how many items → `3`.
- `cities.append("Nakuru")` → stick a new item on the end.

Dictionary cell:
- `customer = { "name": "Stephen", ... }` → pairs of `key: value`.
- `customer["city"]` → fetch the value stored under the key `"city"`.
- `customer["plan"] = "Premium"` → add (or overwrite) the `"plan"` entry.
""")

nb.analogy("A **list** is a numbered row of lockers (open locker #0, #1, #2…). A "
           "**dictionary** is a set of labelled folders (open the folder called "
           "'city'). Use a list when order matters; a dict when names matter.")
nb.warn("Indexing starts at **0**. In a 3-item list the valid positions are 0, 1, 2. "
        "Asking for `cities[3]` errors with 'list index out of range'.")
nb.jargon("list", "an ordered collection accessed by position (index), written with [ ]")
nb.jargon("dictionary", "a collection accessed by name (key), written with { }")
nb.jargon("index", "the position number of an item in a list, starting at 0")
nb.takeaway("Lists = ordered, grab by number `[0]`; dictionaries = labelled, grab by name `[\"key\"]`.")
nb.try_this("Make a dict `car = {\"make\": \"Toyota\", \"year\": 2018}` and print just the year.")

# ---------------------------------------------------------------------------
nb.md("## 0.5 Making decisions — `if` / `elif` / `else`")

nb.plain(r"""
Programs often need to **choose** what to do. `if` runs a block **only when** a
condition is true. It reads like English:

> *if it's raining, take an umbrella; otherwise, wear sunglasses.*

The condition is a yes/no (**boolean**) test. To compare values you use:
`==` (equal?), `!=` (not equal?), `>`, `<`, `>=`, `<=`.

**Indentation matters in Python.** The lines *inside* an `if` are pushed right (4
spaces). That spacing is how Python knows which lines belong to the `if`.
""")

nb.code(r"""
age = 20

if age >= 18:                     # note the colon ':' and the indented line below
    print("Adult")                # runs only if age >= 18 is True
elif age >= 13:                   # 'elif' = "else, if..." checked next
    print("Teenager")
else:                             # if none of the above were True
    print("Child")

# comparison operators produce booleans (True/False):
print("is 5 equal to 5? ", 5 == 5)
print("is 5 equal to 6? ", 5 == 6)
print("is 7 greater than 3?", 7 > 3)
""")

nb.readcode(r"""
- `if age >= 18:` → test "is age at least 18?". The `:` starts the block.
- the **indented** `print("Adult")` → runs only if the test was `True`.
- `elif age >= 13:` → "else, if age is at least 13" — checked only if the first
  test failed. You can have many `elif`s.
- `else:` → the fallback when nothing above matched.
- `5 == 5` → double-equals **asks** "are these equal?" → `True`. (Single `=` would
  try to *assign* and error here.)
""")

nb.warn("`=` assigns, `==` compares. Mixing them up is the most common beginner "
        "error. Read `==` as the question 'are these the same?'.")
nb.jargon("boolean", "a value that is either True or False")
nb.jargon("if statement", "code that runs only when a condition is True")
nb.takeaway("`if/elif/else` picks a branch based on a True/False test; the indented lines are the branch.")

# ---------------------------------------------------------------------------
nb.md("## 0.6 Repeating work — the `for` loop")

nb.plain(r"""
A **loop** does something **once for each item** in a collection — so you don't
copy-paste the same line ten times. A `for` loop reads like:

> *for each city in my list, print a greeting for it.*

This is the backbone of data work: "for each row, do X".
""")

nb.code(r"""
cities = ["Nairobi", "Mombasa", "Kisumu"]

for city in cities:               # 'city' takes each value in turn
    print("Welcome to", city)     # this indented line runs once per city

print("---")

# Loop a fixed number of times with range():
for i in range(3):                # range(3) gives 0, 1, 2
    print("this is loop number", i)
""")

nb.readcode(r"""
- `for city in cities:` → repeat the indented block once for **each** item in
  `cities`. Each time, the variable `city` holds the current item.
- `print("Welcome to", city)` → runs 3 times (once per city).
- `range(3)` → a quick way to get the numbers `0, 1, 2`. `for i in range(3)` loops
  three times, with `i` = 0 then 1 then 2.
""")

nb.deeper(r"""
Later you'll learn that in **NumPy and pandas** we usually *avoid* writing `for`
loops over data, because there's a faster "do it to the whole column at once" way
called **vectorization** (next module). But understanding the loop first is
essential — it's the mental model of "do this for every item", and you'll still use
loops constantly for non-data tasks (files, API calls, experiments).
""")

nb.jargon("for loop", "code that repeats once for each item in a collection")
nb.jargon("range(n)", "produces the numbers 0, 1, ..., n-1 — handy for looping n times")
nb.takeaway("A `for` loop runs the same block once per item — 'for each row, do X'.")
nb.try_this("Loop over `[10, 20, 30]` and print each number doubled.")

# ---------------------------------------------------------------------------
nb.md("## 0.7 Functions — reusable mini-machines")

nb.plain(r"""
A **function** is a **mini-machine**: you feed it **inputs**, it does some work, and
it hands back an **output**. You define it once, then reuse it forever — no copy-paste.

Real-life analogy: a **blender**. You put in fruit (input), press go (the work
inside), and get a smoothie (output). You don't rebuild the blender each time — you
just use it.

We *define* a function with `def`, and *call* (use) it by writing its name with
brackets.
""")

nb.code(r"""
def greet(person):                 # 'def' defines a function called greet
    message = "Hello, " + person   # the work it does (inputs -> some result)
    return message                 # 'return' hands the result back

# Now we CALL it, as many times as we like:
print(greet("Stephen"))
print(greet("Ada"))

def area_of_rectangle(width, height):   # a function can take several inputs
    return width * height

print("area:", area_of_rectangle(4, 3))
""")

nb.readcode(r"""
- `def greet(person):` → define a function named `greet` that expects one input,
  which it will call `person`.
- the indented lines → the work the machine does when called.
- `return message` → send the result back to whoever called the function.
- `greet("Stephen")` → **call** the machine with the input `"Stephen"`; it returns
  `"Hello, Stephen"`, which `print` then shows.
- `area_of_rectangle(4, 3)` → two inputs (`width=4`, `height=3`); returns `12`.
""")

nb.warn("`print` and `return` are different! `print` just *shows* something on "
        "screen. `return` *hands a value back* so other code can use it. A function "
        "that only prints gives you nothing to store in a variable.")
nb.jargon("function", "a reusable mini-machine: inputs go in, an output comes back")
nb.jargon("argument", "an input value you pass into a function")
nb.jargon("return", "the keyword that sends a function's result back to the caller")
nb.takeaway("Define once with `def`, reuse forever by calling it; `return` gives back the result.")

# ---------------------------------------------------------------------------
nb.md("## 0.8 Importing libraries — borrowing other people's code")

nb.plain(r"""
You don't build everything yourself. A **library** is a toolbox of ready-made code
someone else wrote. To use one you **import** it. Data science stands on a few:

- **NumPy** — fast maths on lots of numbers (next module).
- **pandas** — spreadsheets/tables in Python.
- **matplotlib / seaborn** — charts.
- **scikit-learn** — machine-learning models.

We often import "as" a short nickname so we type less (`import pandas as pd`).
""")

nb.code(r"""
import math                       # import the built-in maths library

print("square root of 16:", math.sqrt(16))   # use a tool from it with math.<tool>
print("pi is about:", round(math.pi, 3))

import numpy as np                # import NumPy, nicknamed 'np' (the convention)
print("a NumPy array:", np.array([1, 2, 3]))
""")

nb.readcode(r"""
- `import math` → load the `math` toolbox.
- `math.sqrt(16)` → use the `sqrt` (square-root) tool *from* `math`; the dot means
  "reach inside math and get sqrt". → `4.0`.
- `round(math.pi, 3)` → `math.pi` is the value 3.14159…, and `round(..., 3)` keeps
  3 decimals.
- `import numpy as np` → load NumPy but refer to it by the nickname `np`
  (everyone does this — you'll see `np.` everywhere).
""")

nb.jargon("library", "a toolbox of ready-made code you import to reuse")
nb.jargon("import", "the command that loads a library so you can use its tools")
nb.takeaway("`import name` loads a toolbox; `import name as nick` gives it a short nickname you type instead.")

# ---------------------------------------------------------------------------
nb.md("## 0.9 Errors are normal — how to read them without panic")

nb.plain(r"""
Every programmer — including seniors — sees errors constantly. An error is not a
failure; it's Python **telling you what confused it**, and *where*. The trick is to
read the **last line first** — it names the problem — then look at the line number.
""")

nb.code(r"""
# We deliberately cause an error, then read it. (This cell is MEANT to fail.)
try:
    numbers = [10, 20, 30]
    print(numbers[5])          # position 5 doesn't exist (only 0,1,2)
except Exception as e:
    print("Python raised:", type(e).__name__)
    print("message      :", e)
""")

nb.readcode(r"""
- We ask for `numbers[5]` but the list only has positions 0, 1, 2.
- `try: ... except ...:` lets us *catch* the error so the notebook keeps running and
  we can print it nicely (normally you'd just see a red traceback).
- The important parts of any error:
  - the **type**, here `IndexError` → the *category* of problem.
  - the **message**, here `list index out of range` → the plain explanation.
""")

nb.deeper(r"""
A few error types you'll meet early and what they usually mean:
- **NameError** — you used a variable/function name that doesn't exist (often a typo).
- **TypeError** — you combined incompatible types (e.g. `"5" + 5`).
- **IndexError / KeyError** — you asked for a list position / dict key that isn't there.
- **SyntaxError** — a typo in the code's grammar (missing `:`, bracket, or quote).
- **IndentationError** — your spacing is off (Python cares about indentation).

Workflow: read the **last line**, note the **line number**, fix that one thing, re-run.
Copy-pasting the error message into a search engine is a *normal, professional* move.
""")

nb.jargon("error / exception", "Python's message telling you what confused it and where")
nb.jargon("traceback", "the full error report; read the LAST line first")
nb.takeaway("Errors are guidance, not failure — read the last line, fix that one thing, re-run.")

# ---------------------------------------------------------------------------
nb.md("## 0.10 Putting it together — a tiny real program")

nb.plain(r"""
Let's combine *everything* from this module into one small, readable program: we'll
store some customers, loop over them, make a decision for each, and use a function.
This is the shape of real code — small pieces clicking together.
""")

nb.code(r"""
def risk_label(support_calls):        # a function: input -> output
    if support_calls >= 4:
        return "HIGH risk"
    elif support_calls >= 2:
        return "medium risk"
    else:
        return "low risk"

# a list of dictionaries — a very common data shape (like rows in a table)
customers = [
    {"name": "Ann",  "calls": 5},
    {"name": "Ben",  "calls": 1},
    {"name": "Cara", "calls": 3},
]

for c in customers:                   # loop over each customer (each row)
    label = risk_label(c["calls"])    # call our function on this customer's calls
    print(c["name"], "->", label)
""")

nb.readcode(r"""
- `def risk_label(...)` → a reusable machine that turns a number of support calls
  into a risk word, using `if/elif/else`.
- `customers = [ {..}, {..}, {..} ]` → a **list of dictionaries**: each dict is one
  customer, like one row of a table. (This is *exactly* how tabular data feels
  before we upgrade to pandas.)
- `for c in customers:` → do the block once per customer.
- `c["calls"]` → read that customer's `calls` value (dict lookup by key).
- `risk_label(c["calls"])` → feed it into our function, store the returned word in
  `label`, then print it.
""")

nb.interview(r"""
"I'm comfortable with the Python fundamentals — variables and types, lists and
dicts, control flow with if/for, writing functions, and importing libraries — and I
can read a traceback to debug. That's the base everything else builds on."
""")

nb.md(r"""
## 0.11 Practice (do these before moving on)

1. Make a variable `price = 1200` and print `"The price is 1200"` using commas.
2. Create a list of 4 fruits; print the 1st and the last; add a 5th with `.append`.
3. Build a dict for yourself with keys `name`, `age`, `city`; print each value.
4. Write an `if/elif/else` that prints `"pass"` if a score ≥ 50 else `"retake"`.
5. Loop over `range(5)` and print each number times 10.
6. Write a function `to_celsius(f)` that returns `(f - 32) * 5/9`; call it on `98.6`.
7. Import `math` and print `math.factorial(5)`.
8. Deliberately cause a `NameError` (use an undefined variable), read the message,
   then fix it.

## Summary — the vocabulary you now own

- **variable** (`=`), the four **types** (str/int/float/bool), and `type()`.
- **list** `[ ]` (by index, from 0) and **dictionary** `{ }` (by key).
- **if / elif / else** decisions; comparison operators (`==`, `!=`, `>` …).
- **for** loops ("do X for each item") and `range()`.
- **functions** (`def`, arguments, `return`) — reusable mini-machines.
- **import**ing libraries (and the `as` nickname).
- reading **errors** calmly (last line first).

Next: **Module 1 — Python & NumPy**, where we make Python *fast* on lots of numbers.
""")

out = nb.save("notebooks/0_absolute_basics_python.ipynb",
              glossary_path="notes/glossary.json")
print("saved", out)
