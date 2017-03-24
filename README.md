# lathe

Basic machine learning tools for BYU CS478.

## requirements

- [python2.7 or python3.3+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/) (_optional_)

## installation

```
pip install lathe
```

## usage

```python
import lathe

args = lathe.parse_args()
data, targets = lathe.load(args.file, label_size=args.layers[-1])
```
