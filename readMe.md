# Code Review Agent

- A small Python CLI that reviews a patch, local Git diff, or GitHub pull request.
- Returns possible bugs, risks, missing tests, and style issues with file and line numbers.

## Run the demo

- Requires Python 3.10+. A live review also needs an OpenAI API key.
- Set `OPENAI_API_KEY` in your environment, or put `OPENAI_API_KEY=your-key` in a local `.env` file.
- Set up the project and review the sample patch:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install "openai>=2.11,<3" "httpx>=0.27,<1"
  python review.py --diff examples/buggy.patch
  ```

## Review other changes

- Local working tree: `python review.py --base HEAD --repo /path/to/repo`
- GitHub PR: `python review.py --pr https://github.com/OWNER/REPO/pull/123`
- Public PRs need no GitHub login. Private PRs need a `GITHUB_TOKEN` with read access.
- Add `--json` for JSON output.
- To post one overall review comment on a PR, add `--post` and use a `GITHUB_TOKEN` with Pull requests write access. The CLI posts only when there are findings.

## How it works

- The CLI sends the diff to an OpenAI model and checks finding locations before printing them.
- For local diffs, the model can read a few lines from changed files. For PRs, it reviews the diff only.
- Set `OPENAI_MODEL` or pass `--model` to use another model.

## What I would do next

- Try more public PRs and measure useful findings versus false positives.
- Add optional inline GitHub comments.
