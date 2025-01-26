import os
import subprocess
import sys
from collections import defaultdict

BENCHMARK_KEYWORDS = [
    "math", "bbh", "codex", "gsm", "ifeval", "mbpp", "mmlu", "toxigen", "agi_eval",
    "alpaca_eval", "arc", "boolq", "bigcodebench", "humaneval", "copa", "copycolors",
    "coqa", "cosmosqa", "csqa", "deepmind_math", "drop", "gpqa", "gsm8k", "hellaswag",
    "jeopardy", "logiqa", "minerva_math", "mt_eval", "naturalqs_open", "openbookqa",
    "piqa", "popqa", "sciq", "siqa", "squad", "triviaqa", "truthfulqa", "tydiqa",
    "winogrande", "zero_scrolls"
]

def parse_output_file(output_file):
    """Parses the output.txt file to extract benchmark occurrences."""
    benchmark_occurrences = defaultdict(int)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("  Nombre d'occurrences:"):
                    count = int(line.split(":")[1].strip())
                    benchmark_occurrences[current_benchmark] = count
                elif line.startswith("Mot-clé:"):
                    current_benchmark = line.split("'")[1].strip()
    except Exception as e:
        print(f"Error reading file {output_file}: {e}")
    return benchmark_occurrences

def create_summary_report(base_directory, results_dir, benchmark_results):
    """Creates a summary report of the benchmark analysis."""
    summary_file = os.path.join(results_dir, "benchmark_summary.txt")
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Benchmark Analysis Summary\n")
        f.write("========================\n\n")
        f.write(f"Analyzed Directory: {os.path.basename(base_directory)}\n\n")
        
        f.write("Benchmark Occurrences:\n")
        f.write("--------------------\n")
        sorted_benchmarks = sorted(benchmark_results.items(), key=lambda x: x[1], reverse=True)
        
        for benchmark, count in sorted_benchmarks:
            if count > 0:
                f.write(f"- {benchmark}: {count} occurrences\n")
    
    print(f"Summary report saved to {summary_file}")

def run_benchmark_analysis(base_directory, results_dir):
    """
    Runs keyword analysis for benchmark-related keywords and saves results.
    
    :param base_directory: Directory to analyze
    :param results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)

    output_file = os.path.join(results_dir, "benchmark_results.txt")
    plot_file = os.path.join(results_dir, "benchmark_heatmap.png")

    command = [
        "python", "keywords_occurences_in_repo.py",
        base_directory, *BENCHMARK_KEYWORDS
    ]

    print("Running benchmark keywords analysis...")
    subprocess.run(command)

    if os.path.exists("heatmap.png"):
        os.rename("heatmap.png", plot_file)
        print(f"Saved benchmark heatmap to {plot_file}")
    else:
        print("No heatmap was generated.")

    if os.path.exists("output.txt"):
        os.rename("output.txt", output_file)
        print(f"Saved benchmark results to {output_file}")
        
        benchmark_results = parse_output_file(output_file)
        create_summary_report(base_directory, results_dir, benchmark_results)
    else:
        print("No output file was generated.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_keywords_analysis.py <directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    if not os.path.isdir(base_directory):
        print("Error: The specified directory is not valid.")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "Benchmark")

    run_benchmark_analysis(base_directory, results_dir)

if __name__ == "__main__":
    main()