import datetime
import re

import matplotlib
import networkx as nx
import yaml
from github import Github
import os
from dotenv import load_dotenv, dotenv_values
from urllib.parse import urlparse
import argparse

from action import ActionType, Action
from job import Job
from step import Step
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# App Parameters
parser = argparse.ArgumentParser(description='Get all dependencies of the GitHub workflow from the given GitHub '
                                             'repository')
parser.add_argument('--path', '-P', dest='path', type=str,
                    help='Path of the GitHub repository', required=True)
args = parser.parse_args()

# Environment
config = dotenv_values(".env")
if config['ENVIRONMENT'] == "development":
    load_dotenv('.env.dev')
    GITHUB_ACCESS_TOKEN = os.getenv('GITHUB_ACCESS_TOKEN')

ACTION_PATH = '.github/workflows'


def get_repo_name_from_url(repo_path):
    # Get the name of the repo from an url
    # To modify if ends with .git
    return urlparse(repo_path).path[1:]


def extract_actions_from_file(file, owner_name):
    """
    Extract for all jobs their precedence and steps' actions
    TODO: Add support for versions
    :param file: the workflow file
    :param owner_name: the name of the owner of the repo
    :return: our own representation of the workflow
    """
    try:
        yaml_content = yaml.safe_load(file.decoded_content)
        actions_per_job = {}
        # Loop on every jobs
        for job_name, job_def in yaml_content['jobs'].items():
            actions_per_job[job_name] = []
            job_dependencies = []
            job_steps = []
            # Precedence of the job
            if 'needs' in job_def:
                # handle list of dependencies
                if isinstance(job_def['needs'], list):
                    for dep in job_def['needs']:
                        job_dependencies.append(dep)
                else:
                    job_dependencies.append(job_def['needs'])
            # Loop on every steps
            for steps in job_def['steps']:
                # Loop on every step defs
                for step, step_def in steps.items():
                    # Actions
                    if step == 'uses':
                        action_type = get_action_type(step_def, owner_name)
                        action_obj = Action(step_def, action_type.name)
                        step_obj = Step(action_obj)
                        job_steps.append(step_obj)
            job_obj = Job(job_name, job_dependencies, job_steps)
            actions_per_job[job_name] = job_obj
        return actions_per_job
    except yaml.YAMLError as exc:
        print(exc)


def get_action_type(action_def, owner_name):
    # GitHub action
    if action_def.startswith('actions'):
        return ActionType.GITHUB
    elif action_def.startswith(owner_name):
        return ActionType.PERSONAL
    else:
        return ActionType.PUBLIC


def pretty_print_in_file(filtered_actions, file, repo_name, file_name):
    # Print the result in a file
    file.write(f'name: {repo_name} - {file_name}\n\n')
    for action_type, actions in filtered_actions.items():
        file.write(f'Type: {action_type}\n')
        for action in actions:
            file.write(f'  - {action["node"]} ; utd: {action["up_to_date"]}\n')
    file.write('\n')


"""
Build a graph of the workflow
"""


def dep_graph(jobs):
    graph = {}
    for job_name, job in jobs.items():
        graph[job_name] = []
        for step in job.steps:
            graph[job_name].append(step.action.name)
    return graph


def precedence_graph(jobs):
    graph = {}
    for job_name, job in jobs.items():
        graph[job_name] = []
        for dep in job.dependencies:
            graph[job_name].append(dep)
    return graph


def is_up_to_date(repo_path, version):
    # Check if the workflow is up to date
    g = Github(GITHUB_ACCESS_TOKEN)
    # Retrieve the repo
    repo = g.get_repo(repo_path)
    # get last release
    last_release = repo.get_latest_release()
    # get last release number
    last_release_number = last_release.tag_name

    # set VERSION_TOGGLE based on number of . in version
    if version.count(".") == 1:
        VERSION_TOGGLE = "minor"
    elif version.count(".") == 2:
        VERSION_TOGGLE = "patch"
    else:
        VERSION_TOGGLE = "major"

    print(
        f'\t\tRepo : {repo_path} | Current release: {version} | Last release: {last_release_number}')
    if VERSION_TOGGLE == "major":
        return last_release_number.split('.')[0] == version.split('.')[0]
    elif VERSION_TOGGLE == "minor":
        return last_release_number.split('.')[0] == version.split('.')[0] and last_release_number.split('.')[1] == \
            version.split('.')[1]
    elif VERSION_TOGGLE == "patch":
        return last_release_number.split('.')[0] == version.split('.')[0] and last_release_number.split('.')[1] == \
            version.split('.')[1] and last_release_number.split(
                '.')[2] == version.split('.')[2]


def forked_public_action(repo_path):
    # Check if it forks another repo
    try:
        g = Github(GITHUB_ACCESS_TOKEN)
        # Retrieve the repo
        repo = g.get_repo(repo_path)
        # get parent
        if repo.source is None:
            print(f'\t\tRepo {repo_path} is not a fork')
            return False
        else:
            print(f'\t\tRepo {repo_path} is a fork')
            return forked_public_action(repo.source.full_name)
    except Exception as e:
        print(f'\t\tRepo {repo_path} is not a repo')
        return False


def create_graph_dep(graph=None, file_name=None, safe_users_list=None):
    matplotlib.pyplot.close()
    G = nx.DiGraph()
    for job_name, actions in graph.items():
        G.add_node(job_name)
        for action in actions:
            G.add_edge(job_name, action, weight=2)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    circular = False
    # detect if the same actions are used by different jobs
    for node in G.nodes():
        if G.degree[node] > 1:
            print(f'\t\tAction {node} is used by multiple jobs')
            circular = True
        else:
            print(f'\t\tAction {node} is used by only one job')

    if circular:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)

    # text bottom left corner
    ax.text(0.01, 0.02, "Generated by DirectionMIAGE on " + str(datetime.datetime.now()), wrap=True,
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    plt.figtext(0.5, 0.925,
                "Dependencies " + str(repo_name.split("/")[1]) + " (" + str(repo_name.split("/")[0]) + ")" + "\n" +
                file_name,
                wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')

    # add legend depending on node color
    # Internal > Forked > Truesed > GitHub > Public
    internal_patch = mpatches.Patch(color='#2b3a57', label='Internal')
    forked_patch = mpatches.Patch(color='#b0d4ba', label='Forked')
    trusted_patch = mpatches.Patch(color='#71a36f', label='Trusted')
    skyblue_patch = mpatches.Patch(color='#5e918c', label='GitHub')
    red_patch = mpatches.Patch(color='#945454', label='Public')
    step_patch = mpatches.Patch(color='#cecece', label='Step')

    plt.legend(handles=[internal_patch, forked_patch,
               trusted_patch, skyblue_patch, red_patch, step_patch])

    _github_actions = []
    _public_actions = []
    _internal_actions = []
    _trusted_actions = []
    _forked_actions = []
    _steps = []
    labeldict = {}
    ccls = []

    if len(safe_users_list) > 0:
        ccls.append("The following users are marked as trusted action owners: " +
                    ', '.join(safe_users_list) + ". ")

    for node in G.nodes():
        if node.split('/')[0] == 'actions':
            # get version
            version = node.split('@')[1]
            up_to_date = is_up_to_date(node.split('@')[0], version)
            _github_actions.append(
                {'node': node, 'version': version, 'up_to_date': up_to_date, 'name': node.split('/')[1]})
            labeldict[node] = node.split(
                '/')[1].split('@')[0] + " [" + version + "]"
        elif '@' in node:
            # get version
            version = node.split('@')[1]

            try:
                up_to_date = is_up_to_date(node.split('@')[0], version)
            except:
                print(f'\t\tRepo {node.split("@")[0]} does not exist')
                up_to_date = False

            # verify forks
            if node.split('/')[0] in repo_name.split('/')[0]:
                _internal_actions.append(
                    {'node': node, 'version': version, 'up_to_date': up_to_date, 'name': node.split('/')[1]})
            elif forked_public_action(node.split('@')[0]) is True and (
                    node.split('/')[0] in safe_users_list or node.split('/')[0] in repo_name.split('/')[0]):
                _forked_actions.append(
                    {'node': node, 'version': version, 'up_to_date': up_to_date, 'name': node.split('/')[1]})
            elif node.split('/')[0] in safe_users_list:
                _trusted_actions.append(
                    {'node': node, 'version': version, 'up_to_date': up_to_date, 'name': node.split('/')[1]})
            else:
                ccls.append("The public action \"" + node +
                            "\" is from an untrusted owner.")
                _public_actions.append(
                    {'node': node, 'version': version, 'up_to_date': up_to_date, 'name': node.split('/')[1]})

            labeldict[node] = node.split(
                '/')[1].split('@')[0] + " [" + version + "]"
        elif '@' not in node:
            _steps.append(node)
            labeldict[node] = node

    for action in _github_actions:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[action['node']], node_color='#5e918c',
                                       node_size=[len(action['name']) * 100],
                                       alpha=0.8, label=action['name'], node_shape='s')
        if not action['up_to_date']:
            ccls.append("A newer version of \"" +
                        action['name'].split("@")[0] + "\" is available.")
            nodes.set_edgecolor('red')
            nodes.set_linewidth(3)

    if len(_github_actions) > 0:
        ccls.append("This workflow relies on " +
                    str(len(_github_actions)) + " GitHub-owned actions.")

    for action in _public_actions:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[action['node']], node_color='#945454',
                                       node_size=[len(action['name']) * 100],
                                       alpha=0.8, label=action['name'], node_shape="h")
        if not action['up_to_date']:
            ccls.append("A newer version of \"" +
                        action['name'].split("@")[0] + "\" is available.")
            nodes.set_edgecolor('red')
            nodes.set_linewidth(3)

    for action in _internal_actions:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[action['node']], node_color='#2b3a57',
                                       node_size=[len(action['name']) * 100],
                                       alpha=0.8, label=action['name'], node_shape="s")
        if not action['up_to_date']:
            ccls.append("A newer version of \"" +
                        action['name'].split("@")[0] + "\" is available.")
            nodes.set_edgecolor('red')
            nodes.set_linewidth(3)

    if len(_internal_actions) > 0:
        ccls.append("This workflow relies on " +
                    str(len(_internal_actions)) + " internal actions.")

    for action in _trusted_actions:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[action['node']], node_color='#71a36f',
                                       node_size=[len(action['name']) * 100],
                                       alpha=0.8, label=action['name'], node_shape="h")
        if not action['up_to_date']:
            ccls.append("A newer version of \"" +
                        action['name'].split("@")[0] + "\" is available.")
            nodes.set_edgecolor('red')
            nodes.set_linewidth(3)

    if len(_trusted_actions) > 0:
        ccls.append("This workflow relies on " +
                    str(len(_trusted_actions)) + " actions from trusted owners.")

    for action in _forked_actions:
        nodes = nx.draw_networkx_nodes(G, pos, nodelist=[action['node']], node_color='#b0d4ba',
                                       node_size=[len(action['name']) * 100],
                                       alpha=0.8, label=action['name'], node_shape="s")
        if not action['up_to_date']:
            ccls.append("A newer version of \"" +
                        action['name'].split("@")[0] + "\" is available.")
            nodes.set_edgecolor('red')
            nodes.set_linewidth(3)

    if len(_forked_actions) > 0:
        ccls.append(
            "This workflow relies on " + str(len(_forked_actions)) + " public actions forked by trusted owners.")

    for action in _steps:
        nx.draw_networkx_nodes(G, pos, nodelist=[action], node_color='#cecece',
                               node_size=[len(action) * 100],
                               alpha=0.8, label=action, node_shape='o')

    # set distance between nodes
    # set random edge color for each node
    # get edges
    edges = nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=1)
    for edge in edges:
        edge.set_edgecolor('black')

    nx.draw_networkx_labels(G, pos, font_size=7,
                            font_weight='bold', labels=labeldict)

    # Conclusion
    plt.figtext(0.5, 0.05, ' '.join(ccls), wrap=True, fontweight='bold',
                horizontalalignment='center', fontsize=9)

    if not os.path.exists("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/dependencies"):
        os.makedirs("results/" + research_name_filename + "/" + str(repo_name.split("/")
                    [0]) + "/" + str(repo_name.split("/")[1]) + "/dependencies")

    # save to res
    fig.savefig("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/dependencies/" +
                file_name.split('.')[0] + ".png", dpi=300)

    filtered_actions = {}
    filtered_actions["GITHUB"] = _github_actions
    filtered_actions["INTERNAL"] = _internal_actions
    filtered_actions["PUBLIC"] = _public_actions
    filtered_actions["TRUSTED"] = _trusted_actions
    filtered_actions["FORKED"] = _forked_actions

    return filtered_actions


def create_graph_pre(graph, file_name):
    matplotlib.pyplot.close()
    G = nx.DiGraph()
    for job_name, actions in graph.items():
        G.add_node(job_name)
        for action in actions:
            G.add_edge(job_name, action, weight=2)

    fig, ax = plt.subplots(figsize=(15, 10), dpi=300)

    plt.box(True)

    # text top left corner
    ax.text(0.01, 0.99, "Generated by DirectionMIAGE on " + str(datetime.datetime.now()), wrap=True,
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    plt.figtext(0.5, 0.925, "Precedence " + str(repo_name.split("/")[1]) + " (" + str(repo_name.split("/")[0]) + ")",
                wrap=True, horizontalalignment='center', fontsize=12, fontweight='bold')
    # change distance between nodes
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='#cecece',
                           node_size=1000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', labels={
                            node: node for node in G.nodes()})

    # draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=1)

    if not os.path.exists("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/precedence"):
        os.makedirs("results/" + research_name_filename + "/" + str(repo_name.split("/")
                    [0]) + "/" + str(repo_name.split("/")[1]) + "/precedence")

    # save to res
    fig.savefig(
        "results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/precedence/" + file_name.split(".")[
            0] + ".png", dpi=300)


if __name__ == '__main__':
    # read yml file
    path = args.path
    with open(path, 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)

            research_name = yaml_file['name']
            research_name_filename = re.sub(
                '[^a-zA-Z0-9 \n ]', '', research_name).strip().replace(" ", "_").lower()

            print("*** Starting \"" + research_name + "\" workflow analysis ***")
            for sub_name, sub in yaml_file['subjects'].items():
                matplotlib.pyplot.close()
                # parse content of this yml job

                print("\n\t - Analyzing " + sub_name + " subject(s)...")

                for job in sub['repositories']:
                    matplotlib.pyplot.close()
                    print("\n\t -- Processing " + job + " repository...")
                    repo_name = get_repo_name_from_url(
                        "https://github.com/" + job)

                    # Connect to GitHub
                    try:
                        g = Github(GITHUB_ACCESS_TOKEN)
                        # Retrieve the repo
                        repo = g.get_repo(repo_name)
                        # Retrieve contents from .github/workflow directory
                        contents = repo.get_contents(ACTION_PATH)

                        # create directory if it doesn't exist
                        if not os.path.exists('results'):
                            os.makedirs('results')

                        if not os.path.exists("results/" + research_name_filename):
                            os.makedirs("results/" + research_name_filename)

                        if not os.path.exists("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1])):
                            os.makedirs("results/" + research_name_filename + "/" + str(
                                repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]))

                        if not os.path.exists(
                                "results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/workflows"):
                            os.makedirs(
                                "results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/workflows")

                        actions_per_file = {}
                        while contents:
                            content_element = contents.pop(0)
                            # If dir we extend the research
                            if content_element.type == 'dir':
                                contents.extend(
                                    repo.get_contents(content_element.path))
                            else:
                                print("\t\t - Processing " + content_element.name + " file...")
                                if not content_element.name.endswith(".yml") or content_element.name.endswith(".yaml"):
                                    print("\t\t\t - Skipping " + content_element.name + " file (not a yml file)...")
                                    continue
                                actions_per_file[content_element.name] = extract_actions_from_file(content_element,
                                                                                                   repo.owner.name)
                                # write content_element to file
                                with open("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/workflows/" +
                                          content_element.name, 'w', encoding="utf-8") as f:
                                    f.write(
                                        content_element.decoded_content.decode("utf-8"))

                        file = open("results/" + research_name_filename + "/" + str(repo_name.split("/")[0]) + "/" + str(repo_name.split("/")[1]) + "/" + "findings", "w", encoding="utf-8")
                        for file_name, jobs in actions_per_file.items():
                            graph = precedence_graph(jobs)
                            # print(graph)
                            create_graph_pre(graph, file_name)

                            graph = dep_graph(jobs)
                            # print(graph)
                            filtered_actions = create_graph_dep(graph=graph, file_name=file_name,
                                                                safe_users_list=sub["trusted-owners"] if "trusted-owners" in sub else [])

                            matplotlib.pyplot.close()
                            pretty_print_in_file(filtered_actions, file, repo_name, file_name)

                        matplotlib.pyplot.close()
                        file.close()
                    except Exception as e:
                        raise e
                        print("\t\t - Error: " + str(e))
                        continue

        except yaml.YAMLError as exc:
            print(exc)
