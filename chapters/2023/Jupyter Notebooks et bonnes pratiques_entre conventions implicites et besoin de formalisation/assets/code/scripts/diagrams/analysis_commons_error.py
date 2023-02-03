import json

from scripts.diagrams.extractor import read_files


def get_error_stats(directory: str):
    jsons = read_files(directory)
    error_stats = {}
    for element in jsons:
        try:
            pylint = element['metrics']['code_quality']['pylint_score']['score']
            if pylint == 0:
                # Exclude notebooks where pylint is equals to 0. Not relevant.
                continue

            errors = element['metrics']['code_quality']['pylint_score']['count_by_messages']
            for key, value in errors.items():
                if key not in error_stats:
                    error_stats[key] = value
                else:
                    error_stats[key] += value
        except (TypeError, KeyError):
            pass

    result = dict(sorted(error_stats.items(), key=lambda item: item[1], reverse=True))
    with open(f"commons_errors_{directory.replace('./', '').replace('/', '_')}.json", 'w') as f:
        json.dump(result, f)
    return result


if __name__ == '__main__':
    print(get_error_stats("../../results/"))
