# cs478toolkit

Basic machine learning toolkit for BYU CS478

## requirements

- [python2.7 or python3.3+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/) (_optional_)

## installation 

```
pip install git+https://github.com/mwilliammyers/cs478toolkit
```

## usage

```python
import cs478toolkit

args = cs478toolkit.parse_args()
data, targets = cs478toolkit.load(args.file, label_size=args.layers[-1])
```
