# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

**Getting started**

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/diffrax.git
cd diffrax
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to lint and format the code, and pyright to type-check it.

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

Next verify the tests all pass:

```bash
pip install -r test/requirements.txt
pytest
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
pip install -r docs/requirements.txt
mkdocs build
mkdocs serve
```
(So you run `mkdocs` commands twice; our doc stack is a bit weird and needs this to get the correct module names in the documentation.)

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.
