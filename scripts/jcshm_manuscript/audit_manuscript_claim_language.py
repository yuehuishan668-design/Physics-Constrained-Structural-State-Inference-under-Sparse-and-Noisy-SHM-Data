from pathlib import Path
import re


ROOT = Path(
    "manuscript/jcshm_reconstruction"
)

SEARCH_ROOTS = [
    ROOT / "drafts",
    ROOT / "revision_notes",
]

PATTERNS = {
    "first_claim":
        r"\b(first-ever|for the first time|unprecedented)\b",

    "universal_claim":
        r"\b(universally|universal law|always improves)\b",

    "proof_claim":
        r"\b(proves|proven to|guarantees)\b",

    "upper_bound":
        r"\b(theoretical upper bound|true upper bound)\b",

    "field_claim":
        r"\b(field-ready|field validated|real-world validated)\b",

    "robustness_claim":
        r"\b(robust to measurement noise|noise-robust model)\b",

    "sensor_optimum":
        r"\b(optimal sensor layout|universally optimal)\b",

    "causal_descriptor":
        r"\b(causes prediction failure|causal descriptor)\b",

    "formal_observability":
        r"\b(classical observability|formal identifiability)\b",
}


def main():

    findings = []

    for root in SEARCH_ROOTS:

        if not root.exists():
            continue

        for path in root.rglob("*.md"):

            text = path.read_text(
                encoding="utf-8",
            )

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):

                for label, pattern in PATTERNS.items():

                    if re.search(
                        pattern,
                        line,
                        flags=re.IGNORECASE,
                    ):

                        findings.append(
                            (
                                str(path),
                                line_number,
                                label,
                                line.strip(),
                            )
                        )

    print("=" * 100)
    print("JCSHM MANUSCRIPT CLAIM-LANGUAGE AUDIT")
    print("=" * 100)

    if not findings:

        print("No flagged phrases found.")
        return

    for path, line_number, label, line in findings:

        print()
        print(f"[{label}]")
        print(f"{path}:{line_number}")
        print(line)

    print()
    print("=" * 100)
    print("FLAGGED INSTANCES:", len(findings))
    print("=" * 100)

    print(
        "\nNOTE: A flag is not automatically an error. "
        "Negated/prohibited examples in CORE files are expected."
    )


if __name__ == "__main__":
    main()
