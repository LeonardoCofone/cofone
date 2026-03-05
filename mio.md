# Release Guide

## GitHub only (no PyPI update)
```bash
git add .
git commit -m "descrizione modifica"
git push
```

---

## GitHub + PyPI (nuova versione pubblica)

**1. Aggiorna la versione in `pyproject.toml`**
```toml
version = "0.2.0"
```

**2. Pulisci e rebuilda**
```bash
rm -rf dist/ build/
python -m build
```

**3. Uploada su PyPI**
```bash
python -m twine upload dist/*
# password: pypi-...  (il tuo token)
```

**4. Pusha su GitHub**
```bash
git add .
git commit -m "v0.2.0"
git push
```

---

## Versioning
- `0.1.x` → bug fix
- `0.x.0` → nuove feature
- `x.0.0` → breaking changes