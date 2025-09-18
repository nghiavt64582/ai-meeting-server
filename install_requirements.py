import subprocess
import sys
import re
import os

def install_package(pkg: str):
    """Try to install a single package, return success or fail."""
    try:
        print(f"==> Installing {pkg} ...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ Success: {pkg}")
            return True, ""
        else:
            print(f"❌ Failed: {pkg}")
            return False, result.stdout
    except Exception as e:
        return False, str(e)


def suggest_versions(pkg: str):
    """Try to fetch available versions for a package from pip."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", pkg],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            if lines:
                # Pick first line containing versions
                for line in lines:
                    if "Available versions:" in line:
                        return line.strip()
        return "No version info found."
    except Exception:
        return "Could not fetch versions."


def main(requirements_file):
    if not os.path.exists(requirements_file):
        print(f"File not found: {requirements_file}")
        sys.exit(1)

    failed = []

    with open(requirements_file, "r") as f:
        pkgs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for pkg in pkgs:
        ok, log = install_package(pkg)
        if not ok:
            suggestion = suggest_versions(re.split(r"[=<>]", pkg)[0])
            failed.append((pkg, suggestion))
    
    if failed:
        print("\n=== Failed packages summary ===")
        with open("failed_packages.txt", "w") as out:
            for pkg, suggestion in failed:
                line = f"{pkg} -> {suggestion}"
                print(line)
                out.write(line + "\n")
        print("⚠️ Failed packages logged in failed_packages.txt")
    else:
        print("\n🎉 All packages installed successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python install_requirements.py requirements.txt")
        sys.exit(1)
    main(sys.argv[1])
