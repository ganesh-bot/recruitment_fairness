# import os
import subprocess


def run(cmd):
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# Versioning logic
run("git add .")
run("git commit -m 'Stable base version: preprocessing + testing + model setup'")
run("git tag base-v1")
run("git checkout -b dev-fusion-model")
run("git push origin main --tags")
run("git push origin dev-fusion-model")
