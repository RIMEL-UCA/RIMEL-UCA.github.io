import os
import matplotlib.pyplot as plt


def main():
    data = {}
    dataset_name = ""
    results = os.walk("results")

    for (dir, _, files) in results:
        for file in files:
            if file == "findings":
                # print(f"Processing {file} in {dir}")
                with open(f"{dir}/{file}", "r") as f:
                    dataset_name = dir.split("/")[1]
                    current_file_name = dir.split("/")[2] + "/" + dir.split("/")[3]
                    data[current_file_name] = {}

                    current_ci = None
                    current_type = None

                    for line in f.readlines():
                        # Setup step
                        if line.startswith("name"):
                            current_ci = line.split(" - ")[1].strip()
                            data[current_file_name][current_ci] = []
                            continue

                        if line.startswith("Type: "):
                            current_type = line.split("Type: ")[1].strip()
                            continue

                        if line.startswith("  - "):
                            tmp = line.split("  - ")[1].split(";")
                            action_name = tmp[0].strip()
                            utd = tmp[1].split(":")[1].strip()
                            if action_name not in data[current_file_name][current_ci]:
                                data[current_file_name][current_ci].append(
                                    {'type': current_type, 'action': action_name, 'up_to_date': utd})

    print(f"Dataset: {dataset_name}")
    nb_repo = len(data)
    nb_workflow = sum([len(data[project]) for project in data])
    nb_action = sum([len(data[project][ci]) for project in data for ci in data[project]])
    print(f"\t- {nb_repo} repositories")
    print(f"\t- {nb_workflow} workflows")
    print(f"\t- {nb_action} actions")
    print()
    print(f"\t- {round(nb_action / nb_workflow, 2)} actions/CI workflow avg. [ {nb_action}/{nb_workflow} ]")
    print(f"\t- {round(nb_workflow / nb_repo, 2)} workflows/project avg. [ {nb_workflow}/{nb_repo} ]")
    print(f"\t- {round(nb_action / nb_repo, 2)} actions/project avg. [ {nb_action}/{nb_repo} ]")
    print()

    unsafe_actions = 0
    upgrade_avail_actions = 0
    public_actions = 0
    unsafe_workflows = 0
    unsafe_projects = 0

    percentages = {'GITHUB': 0, 'INTERNAL': 0,
                   'PUBLIC': 0, 'TRUSTED': 0, 'FORKED': 0}

    for project in data:
        for ci in data[project]:
            for action in data[project][ci]:
                percentages[action['type']] += 1

    for project in data:
        unsafeP = False
        for ci in data[project]:
            # print("parsing workflow: ", ci)
            unsafeW = False
            for action in data[project][ci]:
                unsafeA = False
                if action['up_to_date'] == 'False':
                    upgrade_avail_actions += 1
                    unsafeW = True
                    unsafeA = True
                if action['type'] in ['PUBLIC']:
                    public_actions += 1
                    unsafeW = True
                    unsafeA = True

                if unsafeA:
                    unsafe_actions += 1

            if unsafeW:
                unsafe_workflows += 1
                unsafeP = True
        if unsafeP:
            unsafe_projects += 1

    update_and_public = abs(unsafe_actions - (upgrade_avail_actions + public_actions))

    print(
        f"\t- {unsafe_projects} unsafe repositories [ {unsafe_projects}/{nb_repo} — {round(unsafe_projects / nb_repo * 100, 2)}% ]")
    print(
        f"\t- {unsafe_workflows} unsafe workflows [ {unsafe_workflows}/{nb_workflow} — {round(unsafe_workflows / nb_workflow * 100, 2)}% ]")
    print(
        f"\t- {unsafe_actions} unsafe actions [ {unsafe_actions}/{nb_action} — {round(unsafe_actions / nb_action * 100, 2)}% ]")
    print(
        f"\t\t- {upgrade_avail_actions} actions with upgrade available [ {upgrade_avail_actions}/{nb_action} — {round(upgrade_avail_actions / nb_action * 100, 2)}% ]")
    print(
        f"\t\t- {public_actions} public third-party actions [ {public_actions}/{nb_action} — {round(public_actions / nb_action * 100, 2)}% ]")
    print(f"\t- {round(upgrade_avail_actions / unsafe_actions * 100, 2)}% of unsafe actions have an upgrade available")
    print(f"\t- {round(public_actions / unsafe_actions * 100, 2)}% of unsafe actions are public third-party actions")
    print(
        f"\t- {round(update_and_public / public_actions * 100, 2)}% of public actions that need an upgrade [ {update_and_public}/{public_actions} ]")
    print(
        f"\t- {round(unsafe_actions / unsafe_workflows, 2)} unsafe actions/workflow avg. [ {unsafe_actions}/{unsafe_workflows} ]")
    print(
        f"\t- {round(unsafe_workflows / unsafe_projects, 2)} unsafe workflows/project avg. [ {unsafe_workflows}/{unsafe_projects} ]")
    print(
        f"\t- {round(unsafe_actions / unsafe_projects, 2)} unsafe actions/project avg. [ {unsafe_actions}/{unsafe_projects} ]")

    print()
    print(f"\t- {round(unsafe_projects / nb_repo * 100, 2)}% of repositories are unsafe")
    print(f"\t- {round(unsafe_workflows / nb_workflow * 100, 2)}% of workflows are unsafe")
    print(f"\t- {round(unsafe_actions / nb_action * 100, 2)}% of actions are unsafe")

    # plot the reasons for unsafe actions in percentage in a pie chart
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"), dpi=300)
    # the number of unsafe actions that have an upgrade available and that are public third-party actions
    colors = ['#613636', '#854848', '#c99797']
    ax.pie([update_and_public, upgrade_avail_actions - update_and_public, public_actions - update_and_public],
           labels=['outdated & public', 'outdated', 'public'], autopct='%1.1f%%', colors=colors)

    ax.set_title(f"Reasons for unsafe actions")
    plt.show()


    # same thing but counting the safe actions
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"), dpi=300)
    # the number of unsafe actions that have an upgrade available and that are public third-party actions
    colors = ['#5e918c', '#613636', '#854848', '#c99797']
    ax.pie([nb_action - unsafe_actions, update_and_public, upgrade_avail_actions - update_and_public, public_actions - update_and_public],
              labels=['safe', 'outdated & public', 'outdated', 'public'], autopct='%1.1f%%', colors=colors)

    ax.set_title(f"Repartition of the safety of actions")
    plt.show()



    # plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ['#5e918c', '#2b3a57', '#945454', '#71a36f', '#b0d4ba']
    ax.bar(percentages.keys(), percentages.values())

    # set colors for each bar
    for i, bar in enumerate(ax.patches):
        bar.set_color(colors[i])

    ax.set_title(f"Repartition of actions types dataset-wide")
    ax.set_xlabel("Action type")
    ax.set_ylabel("Number of actions")
    plt.show()

    print()
    repartition_per_project = {}

    print(f"\t- repartition of actions types dataset-wide:")
    for key in percentages:
        print(f"\t\t - {round(percentages[key] / nb_action * 100, 2)}% {key} [ {percentages[key]}/{nb_action} ]")
    print()

    print(f"\t- repartition of actions types per project:")
    for project in data:
        # create key
        repartition_per_project[project] = {}
        percentages = {'GITHUB': 0, 'INTERNAL': 0,
                       'PUBLIC': 0, 'TRUSTED': 0, 'FORKED': 0}
        for ci in data[project]:
            for action in data[project][ci]:
                percentages[action['type']] += 1
        #print(f"\t\t- {project}")
        for key in percentages:
            repartition_per_project[project][key] = percentages[key]
            print(f"\t\t\t - {round(percentages[key] / nb_action * 100, 2)}% {key} [ {percentages[key]}/{nb_action} ]")
        #print(repartition_per_project[project])
    # plot the repartition of actions types per project in percentage in a bar chart
    # the format is

    # print(repartition_per_project)
    # create stacked bar char for each project as {'public-apis/public-apis': {'GITHUB': 6, 'INTERNAL': 0, 'PUBLIC': 0, 'TRUSTED': 0, 'FORKED': 0}, 'vercel/next.js': {'GITHUB': 15, 'INTERNAL': 0, 'PUBLIC': 9, 'TRUSTED': 0, 'FORKED': 0}}
    # the format is {'public-apis/public-apis': {'GITHUB': 6, 'INTERNAL': 0, 'PUBLIC': 0, 'TRUSTED': 0, 'FORKED': 0}, 'vercel/next.js': {'GITHUB': 15, 'INTERNAL': 0, 'PUBLIC': 9, 'TRUSTED': 0, 'FORKED': 0}}

    labels = ['GITHUB', 'INTERNAL', 'PUBLIC', 'TRUSTED', 'FORKED']
    projects = repartition_per_project.keys()
    # to array
    projects = list(projects)

    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    list_github = []
    list_internal = []
    list_public = []
    list_trusted = []
    list_forked = []

    for project in projects:
        list_forked.append(repartition_per_project[project]['FORKED'])
        list_trusted.append(repartition_per_project[project]['TRUSTED'])
        list_public.append(repartition_per_project[project]['PUBLIC'])
        list_internal.append(repartition_per_project[project]['INTERNAL'])
        list_github.append(repartition_per_project[project]['GITHUB'])

    ax.bar(projects, list_github , label="GitHub", color='#5e918c')
    ax.bar(projects, list_internal, bottom=list_github, label="Internal", color='#2b3a57')
    ax.bar(projects, list_public, bottom=list_github, label="Public", color='#945454')
    ax.bar(projects, list_trusted, bottom=list_github, label="Trusted", color='#71a36f')
    ax.bar(projects, list_forked, bottom=list_github, label="Forked", color='#b0d4ba')

    ax.set_title(f"Repartition of actions types per project")
    ax.set_xlabel("Repository")
    ax.set_ylabel("Number of actions")
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    ax.legend()
    plt.show()

    # list all project separated by a comma
    projects = list(repartition_per_project.keys())
    projects = ", ".join(projects)
    print(projects)

    # generate a .md file with the results, ordered by project
    with open("results.md", "w") as f:
        f.write(f"# {dataset_name}\n\n")
        f.write(f"\n## Corpus\n\n")
        f.write(f"The corpus contains {nb_action} actions from {len(data)} projects: ")
        f.write(f"{projects}\n")

        f.write(f"## Repartition of actions types dataset-wide\n\n")
        for key in percentages:
            f.write(f"- {round(percentages[key] / nb_action * 100, 2)}% {key} [ {percentages[key]}/{nb_action} ]\n")
        f.write(f"\n## Projects\n\n")
        for project in data:
            f.write(f"### {project}\n\n")
            f.write(f"#### Repartition of actions types\n\n")
            # table markdown
            f.write(f"| Action type | Percentage | Number of actions |\n")
            f.write(f"| --- | --- | --- |\n")
            for key in percentages:
                f.write(f"| {key} | {round(repartition_per_project[project][key] / nb_action * 100, 2)}% | {repartition_per_project[project][key]} |\n")
            f.write(f"\n#### List of actions\n\n")
            f.write(f"| Action type | Action name | Up to date |\n")
            f.write(f"| --- | --- | --- |\n")
            for ci in data[project]:
                for action in data[project][ci]:
                    f.write(f"| {action['type']} | {action['action']} | {action['up_to_date']} |\n")

            f.write(f"\n#### Precedence\n\n")

            # for each workflow in project
            for ci in data[project]:
                f.write(f"![Precedence {ci}](results/{dataset_name}/{project}/precedence/{ci.replace('.yml','').replace('.yaml','')}.png)\n")

            f.write("\n")

            f.write(f"\n#### Dependencies\n\n")

            # for each workflow in project
            for ci in data[project]:
                f.write(f"![Dependencies {ci}](results/{dataset_name}/{project}/dependencies/{ci.replace('.yml','').replace('.yaml','')}.png)\n")

if __name__ == '__main__':
    main()
