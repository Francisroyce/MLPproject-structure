# src/MLP_project_structure/cli.py

import argparse

def main():
    parser = argparse.ArgumentParser(description="MLP CLI Tool")
    parser.add_argument("--greet", type=str, help="Your name to greet")

    args = parser.parse_args()

    if args.greet:
        print(f"Hello, {args.greet}! Welcome to the MLP CLI.")
    else:
        print("Welcome to the MLP CLI. Use --greet to receive a personalized greeting.")
