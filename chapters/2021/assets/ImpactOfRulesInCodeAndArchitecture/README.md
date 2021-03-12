# rimel-si5

## Graphs

[Results](https://docs.google.com/spreadsheets/d/1Kp7109neJxEUN6DTmd7o7k75qvT8_GvHvgbxMcl-Z4w/edit?usp=sharing)

## Raw data

All raw data are in the `results` directory.

## Repositories

You can find a zip file with the clone of all the repositories with which we performed our study in the `repositories` folder.

## Input files

You can find all our input files in the `input_files` folder.

## How to run

### Occurence finder in a directory

```bash
py main.py <directory>
```

_The `input.txt` file with the keywords need to be in `input_files` directory._

### Git extractor

To run the git extractor:

```bash
py git_extractor.py <input_file> <MODE>
```

_The `input_file` should look like the `input_files/github.txt` or `input_files/gitlab.txt`. Where the first argument of the line is the owner of the repository and the second argument (after the comma), is the repository name._
_The `MODE` is `github` to extract data with github API and `gitlab` to extract data with gitlab API._
