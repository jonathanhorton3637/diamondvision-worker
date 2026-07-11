"""DiamondVision Worker deployment helper."""

from __future__ import annotations

from datetime import datetime
import getpass
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile


REPOSITORY = Path("/workspace/diamondvision-worker")
REMOTE = "origin"
BRANCH = "main"


def git(*arguments: str, check: bool = True, env=None):
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        env=env,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="")

    if check and result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or f"Git command failed: {' '.join(arguments)}"
        )

    return result


def remove_temporary_files() -> None:
    for folder in REPOSITORY.rglob("__pycache__"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)

    for file in REPOSITORY.rglob("*.pyc"):
        file.unlink(missing_ok=True)


def create_askpass(folder: Path) -> Path:
    script = folder / "github-askpass.sh"

    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  *Username*) printf "%s\\n" "$GITHUB_USERNAME" ;;\n'
        '  *) printf "%s\\n" "$GITHUB_TOKEN" ;;\n'
        "esac\n",
        encoding="utf-8",
    )

    script.chmod(
        script.stat().st_mode | stat.S_IXUSR
    )

    return script


def main() -> None:
    print()
    print("=" * 60)
    print("DiamondVision Worker Deployment")
    print("=" * 60)

    if not (REPOSITORY / ".git").is_dir():
        raise RuntimeError(
            f"Git repository not found: {REPOSITORY}"
        )

    remove_temporary_files()

    print("\nCurrent changes:\n")
    status = git("status", "--short")
    has_changes = bool(status.stdout.strip())

    if has_changes:
        default_message = (
            "DiamondVision Worker update "
            + datetime.now().strftime("%Y-%m-%d %H:%M")
        )

        message = input(
            f"\nCommit message [{default_message}]: "
        ).strip() or default_message

        git("add", "--all")

        staged = git(
            "diff",
            "--cached",
            "--quiet",
            check=False,
        )

        if staged.returncode == 1:
            git("commit", "-m", message)

        elif staged.returncode != 0:
            raise RuntimeError(
                "Unable to inspect staged changes."
            )

    else:
        print("No local changes need to be committed.")

    print("\nFetching GitHub...")
    git("fetch", REMOTE)

    print("\nRebasing onto origin/main...")
    rebase = git(
        "rebase",
        f"{REMOTE}/{BRANCH}",
        check=False,
    )

    if rebase.returncode != 0:
        print()
        print("Git found a conflict.")
        print("Resolve the conflict, then run:")
        print("  git add <resolved-file>")
        print("  GIT_EDITOR=true git rebase --continue")
        print("  python3 deploy.py")
        raise SystemExit(1)

    username = input("\nGitHub username: ").strip()

    if not username:
        raise ValueError("GitHub username cannot be empty.")

    token = getpass.getpass(
        "GitHub Personal Access Token: "
    ).strip()

    if not token:
        raise ValueError(
            "GitHub Personal Access Token cannot be empty."
        )

    remote_url = git(
        "remote",
        "get-url",
        REMOTE,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(
        prefix="diamondvision-git-"
    ) as temporary:
        askpass = create_askpass(Path(temporary))

        environment = os.environ.copy()
        environment["GIT_ASKPASS"] = str(askpass)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GITHUB_USERNAME"] = username
        environment["GITHUB_TOKEN"] = token

        print("\nPushing to GitHub...")

        git(
            "-c",
            "credential.helper=",
            "push",
            remote_url,
            f"HEAD:{BRANCH}",
            env=environment,
        )

    print()
    print("=" * 60)
    print("Deployment push completed successfully.")
    print("GitHub Actions should now rebuild the worker.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDeployment cancelled.")
        raise SystemExit(130)
    except Exception as error:
        print(f"\nDeployment failed: {error}")
        raise SystemExit(1)
