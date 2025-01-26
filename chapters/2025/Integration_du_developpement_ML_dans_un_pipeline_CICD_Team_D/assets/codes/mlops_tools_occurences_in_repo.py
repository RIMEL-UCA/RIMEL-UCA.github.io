import os
import subprocess
import sys
from collections import defaultdict

CATEGORIES = {
    "Tools ML": [
        "mlops", "openml", "clearml", "mlflow", "kitops", "wandb", "mlreef", "valohai",
        "hopsworks", "polyaxon", "mlrun", "iguazio", "databricks", "pachyderm", "alteryx",
        "domino", "h2o.ai", "sas"
    ],
    "Data Management - Data exploration & Management": [
        "alluxio", "amundsen", "aparavi", "atscale", "cloudera", "datagrok", "dataiku",
        "delta", "datera", "dremio", "elastifile", "erwin", "excelero", "flur", "geminidata",
        "hammerspace", "hudi", "hycu", "imply", "komprise", "kyvosinsights", "milvus",
        "octopai", "pilosa", "prestodb", "qri", "rubrik", "spark", "tamr", "vearch", "vexata",
        "yellowbrick"
    ],
    "Data Management - Data labeling": [
        "appen", "dataturks", "definedcrowd", "doccano", "imerit", "labelbox", "prodi",
        "playment", "scale", "segments", "snorkel", "supervisely"
    ],
    "Data Management - Data streaming": [
        "kinesis", "aresdb", "confluent", "dataflow", "hudi", "flink", "kafka", "storm",
        "striim", "valohai"
    ],
    "Data Management - Data versioning": [
        "dagshub", "databricks", "dataiku", "dolthub", "dvc", "floydhub", "mlreef", "pachyderm",
        "qri", "waterline"
    ],
    "Data Management - Data privacy": [
        "amnesia", "aircloak", "dataanon", "celantur", "pysyft", "tumult"
    ],
    "Data Management - Data quality": [
        "arize", "naveego"
    ],
    "Data Management - Data processing and visualization": [
        "alteryx", "ascend", "dask", "dataiku", "databricks", "dotdata", "flyte", "gluent",
        "koalas", "iguazio", "imply", "incorta", "mlflow", "kyvosinsights", "mlreef", "modin",
        "naveego", "openml", "pachyderm", "pilosa", "prestodb", "sas", "snorkel", "sqlflow",
        "starburst", "turicreate", "vaex", "valohai", "wandb"
    ],
    "Data Management - Feature engineering": [
        "dotdata", "feast", "featuretools", "pachyderm", "scribbledata", "tecton", "tsfresh",
        "mlreef", "iguazio"
    ],
    "Data Management - Model Training": [
        "alteryx", "iguazio", "colab", "databricks", "dataiku", "domino", "dotscience",
        "floydhub", "flyte", "horovod", "ludwig", "kaggle", "mlreef", "h2o", "metaflow",
        "mlflow", "paperspace", "perceptilabs", "snorkel", "turicreate", "valohai", "sas",
        "anyscale", "pachyderm"
    ],
    "Data Management - Experiment tracking": [
        "allegro", "comet", "dagshub", "dataiku", "datarobot", "datmo", "domino", "floydhub",
        "h2o", "ludwig", "iguazio", "mlflow", "mlreef", "neptune", "openml", "polyaxon",
        "spell", "valohai", "wandb"
    ],
    "Data Management - Model / Hyperparameter optimization": [
        "alteryx", "angel", "comet", "datarobot", "hyperopt", "polyaxon", "sigopt", "tune",
        "optuna", "talos"
    ],
    "Data Management - Model management": [
        "algorithmia", "allegro", "databricks", "dataiku", "determined", "dockship", "domino",
        "dotdata", "floydhub", "gluon", "h2o", "huggingface", "watson", "iguazio", "mlflow",
        "modzy", "perceptilabs", "sas", "turicreate", "valohai", "verta"
    ],
    "Data Management - Model evaluation": [
        "arize", "dawnbench", "mlperf", "streamlit", "tensorboard"
    ],
    "Data Management - Model explainability": [
        "fiddler", "interpretml", "lucid", "perceptilabs", "shap", "tensorboard"
    ],
    "CD - Data flow management": [
        "alluxio", "spark", "ascend", "kafka", "dataiku", "dotdata", "hycu"
    ],
    "CD - Feature transformation": [
        "feast", "featuretools", "scribbledata", "tecton", "iguazio"
    ],
    "CD - Monitoring": [
        "algorithmia", "arize", "dataiku", "datadog", "datatron", "datarobot", "domino",
        "dotscience", "h2o", "iguazio", "losswise", "snorkel", "unravel", "valohai", "verta"
    ],
    "CD - Model Compliance & Audit": [
        "algorithmia", "sas", "h2o"
    ],
    "CD - Model Deployment & Serving": [
        "aible", "algorithmia", "allegro", "clipper", "coreml", "cortex", "dataiku",
        "datatron", "datmo", "domino", "dotdata", "dotscience", "floydhub", "fritz", "watson",
        "iguazio", "kubeflow", "mlflow", "modzy", "octoml", "paperspace", "prediction", "sas",
        "seldon", "streamlit", "h2o", "valohai", "verta"
    ],
    "CD - Model validation": [
        "arize", "datatron", "fiddler", "lucid", "mlperf", "sas", "streamlit"
    ],
    "Computing Management - Computing and data infrastructure (servers)": [
        "watson", "aws", "azure", "cloudera", "paperspace", "kamatera", "linode",
        "cloudways", "liquidweb", "digitalocean", "vultr"
    ],
    "Computing Management - Environment Management": [
        "conda", "databricks", "datmo", "mahout", "mlreef"
    ],
    "Computing Management - Resource allocation": [
        "sagemaker", "algorithmia", "mlreef", "databricks", "azure", "dataiku",
        "determinedai", "floydhub", "watson", "polyaxon", "spellml", "valohai", "allegro"
    ],
    "Computing Management - Scaling": [
        "sagemaker", "argo", "azure", "datadog", "datatron", "datmo", "tensorrt", "seldon", "tvm"
    ],
    "Computing Management - Security & privacy": [
        "algorithmia", "cleverhans", "datadog", "modzy", "pysyft", "tumult"
    ]
}

def run_keyword_analysis(base_directory, results_dir):
    """Runs keyword analysis for each category and saves results."""
    for category, keywords in CATEGORIES.items():
        category_dir = os.path.join(results_dir, category.replace(" ", "_"))
        os.makedirs(category_dir, exist_ok=True)

        output_file = os.path.join(category_dir, "output.txt")
        plot_file = os.path.join(category_dir, "heatmap.png")

        command = [
            "python", "keywords_occurences_in_repo.py",
            base_directory, *keywords
        ]

        print(f"Running analysis for category: {category}")
        subprocess.run(command)

        if os.path.exists("heatmap.png"):
            os.rename("heatmap.png", plot_file)
            print(f"Saved heatmap for category '{category}' to {plot_file}")
        else:
            print(f"No heatmap generated for category '{category}'.")

        if os.path.exists("output.txt"):
            os.rename("output.txt", output_file)
            print(f"Saved output for category '{category}' to {output_file}")
        else:
            print(f"No output generated for category '{category}'.")

def parse_output_file(output_file):
    """Parses the output.txt file to extract tool occurrences."""
    tool_occurrences = defaultdict(int)
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("  Nombre d'occurrences:"):
                    count = int(line.split(":")[1].strip())
                    tool_occurrences[current_tool] += count
                elif line.startswith("Mot-clé:"):
                    current_tool = line.split("'")[1].strip()
    except Exception as e:
        print(f"Error reading file {output_file}: {e}")
    return tool_occurrences

def summarize_findings(base_directory, results_dir):
    """Summarizes the findings by analyzing all output.txt files."""
    tool_summary = defaultdict(lambda: {"count": 0, "categories": set()})
    
    for category_dir in os.listdir(results_dir):
        category_path = os.path.join(results_dir, category_dir)
        if os.path.isdir(category_path):
            output_file = os.path.join(category_path, "output.txt")
            if os.path.exists(output_file):
                tool_occurrences = parse_output_file(output_file)
                for tool, count in tool_occurrences.items():
                    tool_summary[tool]["count"] += count
                    tool_summary[tool]["categories"].add(category_dir.replace("_", " "))

    summary_file = os.path.join(results_dir, "summary_report.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Summary of Findings\n")
        f.write("===================\n\n")
        f.write(f"Analyzed Directory: {base_directory}\n\n")

        f.write("Most Used Tools:\n")
        f.write("----------------\n")
        sorted_tools = sorted(tool_summary.items(), key=lambda x: x[1]["count"], reverse=True)
        for tool, data in sorted_tools:
            categories = ", ".join(data["categories"])
            f.write(f"- {tool}: {data['count']} occurrences (Categories: {categories})\n")
    
    print(f"Summary report saved to {summary_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python wrapper_script.py <directory>")
        sys.exit(1)

    base_directory = sys.argv[1]
    if not os.path.isdir(base_directory):
        print("Error: The specified directory is not valid.")
        sys.exit(1)

    project_name = os.path.basename(base_directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, f"results_{project_name}")
    os.makedirs(results_dir, exist_ok=True)

    run_keyword_analysis(base_directory, results_dir)

    summarize_findings(base_directory, results_dir)

if __name__ == "__main__":
    main()
