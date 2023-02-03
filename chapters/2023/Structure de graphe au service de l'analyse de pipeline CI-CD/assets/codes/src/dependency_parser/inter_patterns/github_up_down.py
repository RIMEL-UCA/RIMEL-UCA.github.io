from src.dependency_parser.utils import steps, job_names
from src.dependency_parser.strategies import InterScanStrategy


class Strategy(InterScanStrategy):
    @staticmethod
    def __get_download_dependencies(steps: list):
        return [
            step['with']['name'] if 'name' in step['with'] else '??'
            for step in steps if 'uses' in step and 'actions/download-artifact' in step['uses']
        ]

    @staticmethod
    def __get_uploaded_artifacts(steps: list):
        return [
            step['with']['name'] if 'name' in step['with'] else '??'
            for step in steps if 'uses' in step and 'actions/upload-artifact' in step['uses']
        ]


    @staticmethod
    def __get_possible_uploaders(artifact: str, applicants: dict):
        return [name for name, data in applicants.items() if artifact in data['uploads']]

    @staticmethod
    def __reduce_github_vars(artifact_name: str):
        return artifact_name.replace('${{ github.', '{').replace(' }}', '}')
    
    @staticmethod
    def __add_dep(uploader, downloader, message, graph: dict):
        deps = []
        if uploader in graph:
            deps = graph[uploader]
        else:
            graph[uploader] = deps
        
        deps += [(downloader, 'up/down', message)]

    def parse(self, jobs: dict, graph: dict):
        job_data = {}

        for name, job in jobs.items():
            job_data[name] = {
                # Which atifacts are needed by the job
                'downloads': self.__get_download_dependencies(job['steps']),
                # # Which jobs must be run before this job
                'needs': job['needs'] if 'needs' in job else [],
                # Which atifacts are produced by the job
                'uploads': self.__get_uploaded_artifacts(job['steps'])
            }
        
        for name, data in job_data.items():
            for dl_dep in data['downloads']:
                uploaders = self.__get_possible_uploaders(dl_dep, job_data)
                dep = self.__reduce_github_vars(dl_dep)

                if not uploaders:
                    self.__add_dep('?', name, dep, graph)

                for uploader in uploaders:
                    if uploader in data['needs']:
                        self.__add_dep(uploader, name, dep, graph)
                    else:
                        self.__add_dep(uploader, name, f"{dep} (need missing)", graph)


def download_link(linked_dep, yml, current_job, action_step):

    download_dep = action_step['with']['name']

    #"If the download dependency is not satisfied, this may be due to an unknown artefact"
    if download_dep not in linked_dep.keys():
        linked_dep[download_dep] = {
            'producer': 'Unknown',
            'consumer': [current_job],
            'producer_inconsistency': [
                {'warning': 'orange', 'problem': f"Artifacts {download_dep} not declared : "}
            ]
        }
    elif 'needs' in yml['jobs'][current_job] and linked_dep[download_dep]['producer'] in yml['jobs'][current_job]['needs']:
        linked_dep[download_dep]['consumer'] += [current_job]
    # In case needs are missing.
    else:
        linked_dep[download_dep]['consumer'] += [current_job]
        linked_dep[download_dep]['consumer_inconsistency'] = [
            {'warning': 'orange', 'problem': f"Missing Needs for {current_job} : " + linked_dep[download_dep]['producer']}
        ]



def upload_link(linked_dep, yml, job, action):
    """Find out every upload pattern, and store the upload as producer job"""
    linked_dep[action['with']['name']] = {'producer': job, 'consumer': []}


pattern = {
    'actions/upload': upload_link,
    'actions/download': download_link,
}


def detect(yml):
    """find if there is any pattern that match with the current pipeline checked."""
    # verify that a step match with an upload or download pattern
    def match_with_pattern(step):
        return list(filter(lambda key: key in step['uses'], pattern.keys()))

    # an array of tuples that contains all steps that match with a pattern, (is_dep,relation object)
    def matching_steps(yml, job):
        _steps = {patternKey: [] for patternKey in pattern.keys()}
        is_dep = False
        for step in steps(yml, job):
            if 'uses' not in step or len((match := match_with_pattern(step))) == 0: continue
            _steps[match[0]] += [step]
            is_dep = True

        return (is_dep, _steps)

    # a map which contains as key job names and as values : steps that contains a composite pattern action : upload or download.
    job_map = {job: match[1] for job in job_names(yml) if ((match := matching_steps(yml, job))[0])}

    return job_map


def links_dependencies(yml, job_map):
    """Run every pattern and add a new linked dependency into the linked_dep."""
    linked_dep = {}
    for job, matching_steps in job_map.items():
        for pattern_name, actions in matching_steps.items():
            for action in actions:
                pattern[pattern_name](linked_dep, yml, job, action)
    return linked_dep


def build_graph(dot, dep):
    for artefact in dep.keys():
        # append the artefact creator
        producer = dep[artefact]['producer']
        dot.node(producer)

        if 'consumer_inconsistency' in dep[artefact]:
            for inconsistency in dep[artefact]['consumer_inconsistency']:
                for consumer in dep[artefact]['consumer']:
                    dot.edge(producer, consumer, label=inconsistency['problem'], color=inconsistency['warning'])

        elif 'producer_inconsistency' in dep[artefact]:
            for inconsistency in dep[artefact]['producer_inconsistency']:
                for consumer in dep[artefact]['consumer']:
                    dot.edge(dep[artefact]['producer'], consumer, label=inconsistency['problem'],
                         color=inconsistency['warning'])
        # Normal case
        else:
            for consumer in dep[artefact]['consumer']:
                dot.node(consumer)
                dot.edge(dep[artefact]['producer'], consumer, label=artefact)
