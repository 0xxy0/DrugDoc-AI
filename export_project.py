from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "drugdoc-ai_code.md"

# Directories to ignore
IGNORED_DIRS = {
    ".git",
    ".github",  # Remove this if you want to include GitHub workflows
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "ENV",
    "node_modules",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "htmlcov",
    "coverage",
    "dist",
    "build",
    "site-packages",
    ".next",
    ".nuxt",
    ".turbo",
    "out",
    "target",
    "vendor",
}

# Files to ignore
IGNORED_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".coverage",
    "coverage.xml",
    "npm-debug.log",
    "yarn-debug.log",
    "yarn-error.log",
    "pnpm-debug.log",
}

# File patterns to ignore
IGNORED_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.class",
    "*.o",
    "*.a",
    "*.log",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*.swo",
    "*~",
}

# Sensitive files whose contents should never be exported
SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
}

# Large or generated files that should not be embedded
MAX_FILE_SIZE = 1_000_000  # 1 MB

# Extensions and their Markdown code-block languages
LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".json": "json",
    ".jsonc": "json",
    ".md": "markdown",
    ".mdx": "mdx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "text",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ps1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".txt": "text",
    ".xml": "xml",
    ".svg": "xml",
    ".env.example": "dotenv",
    ".dockerfile": "dockerfile",
    ".ipynb": "json",
    ".r": "r",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".prisma": "prisma",
    ".tf": "hcl",
    ".make": "makefile",
}

# Common filenames without conventional extensions
SPECIAL_LANGUAGES = {
    "Dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "Makefile": "makefile",
    "Justfile": "make",
    "Procfile": "text",
    "requirements.txt": "text",
    "Pipfile": "toml",
    "Pipfile.lock": "json",
    "poetry.lock": "text",
    "uv.lock": "text",
    "package-lock.json": "json",
    "pnpm-lock.yaml": "yaml",
    "yarn.lock": "yaml",
    ".gitignore": "gitignore",
    ".dockerignore": "text",
    ".prettierrc": "json",
    ".eslintrc": "json",
}


def should_ignore(path: Path) -> bool:
    """Return True if a file or directory should be excluded."""
    if any(part in IGNORED_DIRS for part in path.parts):
        return True

    if path.name in IGNORED_FILES:
        return True

    if path.name in SENSITIVE_FILES:
        return True

    if path.name.startswith(".env.") and path.name != ".env.example":
        return True

    if any(path.match(pattern) for pattern in IGNORED_PATTERNS):
        return True

    return False


def visible_entries(path: Path):
    """Return sorted project entries, excluding ignored items."""
    return sorted(
        (
            entry
            for entry in path.iterdir()
            if not should_ignore(entry)
        ),
        key=lambda entry: (entry.is_file(), entry.name.lower()),
    )


def build_tree(path: Path, prefix: str = "") -> list[str]:
    """Build a visual project tree."""
    lines = []
    entries = visible_entries(path)

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1
        connector = "└── " if is_last else "├── "

        lines.append(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(
                build_tree(entry, prefix + extension)
            )

    return lines


def language_for_file(path: Path) -> str:
    """Determine the Markdown code-block language."""
    if path.name in SPECIAL_LANGUAGES:
        return SPECIAL_LANGUAGES[path.name]

    if path.suffix.lower() in LANGUAGES:
        return LANGUAGES[path.suffix.lower()]

    # Handle names such as .env.example
    for suffix, language in LANGUAGES.items():
        if path.name.endswith(suffix):
            return language

    return "text"


def collect_files(path: Path) -> list[Path]:
    """Collect eligible files without traversing ignored directories."""
    files = []

    def walk(directory: Path):
        for entry in visible_entries(directory):
            if entry.is_dir():
                walk(entry)
            elif entry.is_file():
                files.append(entry)

    walk(path)
    return sorted(files, key=lambda p: p.relative_to(path).as_posix().lower())


def safe_read_text(file: Path) -> str | None:
    """Read a text file, skipping binary and oversized files."""
    try:
        if file.stat().st_size > MAX_FILE_SIZE:
            return None

        return file.read_text(encoding="utf-8")

    except (UnicodeDecodeError, OSError):
        return None


def export_project():
    """Export the project tree and eligible file contents to Markdown."""
    if not ROOT.exists():
        raise FileNotFoundError(
            f"Project directory not found: {ROOT.resolve()}\n"
            "Place this script beside the drugdoc-ai directory."
        )

    if not ROOT.is_dir():
        raise NotADirectoryError(
            f"Project path is not a directory: {ROOT.resolve()}"
        )

    tree = build_tree(ROOT)
    files = collect_files(ROOT)

    with OUTPUT.open("w", encoding="utf-8") as output:
        output.write(f"# {ROOT.name} — Complete Project Export\n\n")

        output.write("## Project Structure\n\n")
        output.write("```text\n")
        output.write(f"{ROOT.name}/\n")

        if tree:
            output.write("\n".join(tree) + "\n")

        output.write("```\n\n")

        output.write("## File Contents\n\n")

        included = 0
        skipped = 0

        for file in files:
            relative_path = file.relative_to(ROOT).as_posix()
            content = safe_read_text(file)

            output.write(f"### `{relative_path}`\n\n")

            if content is None:
                output.write(
                    "*Contents omitted: binary, unreadable, or larger "
                    f"than {MAX_FILE_SIZE:,} bytes.*\n\n"
                )
                skipped += 1
                continue

            language = language_for_file(file)

            # Use a fence longer than any backtick run in the source,
            # so embedded Markdown cannot break the export.
            longest_run = max(
                (
                    len(run)
                    for run in content.split("```")
                    if run
                ),
                default=0,
            )
            fence = "`" * max(3, longest_run + 1)

            output.write(f"{fence}{language}\n")
            output.write(content)

            if not content.endswith("\n"):
                output.write("\n")

            output.write(f"{fence}\n\n")
            included += 1

    print(f"Export created: {OUTPUT.resolve()}")
    print(f"Files included: {included}")
    print(f"Files omitted:  {skipped}")
    print(f"Project tree entries: {len(tree)}")


if __name__ == "__main__":
    export_project()