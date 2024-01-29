from github import Github
import pandas as pd
from utils.docker_compose_analyser import Docker_compose_analyser
from utils.individualdeployment import individual_deployment
from utils.mongo_analyse import mongo_analyzer
from utils.master_slave import masterslave_analyzer
from utils.events_analyze import event_analyser
from utils.load_balacing import loadbalancer_analyzer
from utils.CI_CD_analyze import cicd_analyzer
from utils.gateway import gateway_analyzer
from utils.db_analyser.db_analyser import DB_analyser
from utils.check_microservice import microservice_keywords
from utils.Colors import Couleurs
import argparse


def analyze_repository(repository, results_df, token):
    print(Couleurs.BLEU+"\n\n======================================================\n")
    print(repository)
    print("\n======================================================\n\n"+Couleurs.RESET)

    # check docker compose
    print("\n\n====================================================== "+Couleurs.VERT+"HAS DOCKER COMPOSE"+Couleurs.RESET+"\n")
    analyse = Docker_compose_analyser()
    dockercompose = analyse.has_docker_compose(repository=repository)
    docker_compose_status = "Present" if dockercompose is not None else "Not"
    print("[ "+Couleurs.VERT+"DOCKER COMPOSE"+Couleurs.RESET+" ] : ", docker_compose_status)

    if dockercompose:
        ## check db
        print("\n\n====================================================== "+Couleurs.VERT+"CHECK DBs"+Couleurs.RESET+"\n")
        db_analyser = DB_analyser(token)
        db_analyser_result = db_analyser.run(repository=repository, dockercompose=dockercompose)
        db_analyser_status = "Not"
        if db_analyser_result == 1:
            db_analyser_status = "Present"
        elif db_analyser_result == -1:
            db_analyser_status = "Need for observation"
        print("[ " + Couleurs.VERT + "DBs" + Couleurs.RESET + " ] : ", db_analyser_status)

        ## get all directories
        print("\n\n====================================================== "+Couleurs.VERT+"GET ALL DIRECTORIES"+Couleurs.RESET+"\n")
        directories = analyse.get_all_directories(repository=repository, path="")
        images = analyse.get_services_from_docker_compose(repository=repository, dockercompose=dockercompose)
        print("[ " + Couleurs.VERT + "IMAGES" + Couleurs.RESET + " ] : ", images)

        # check custom images
        print("\n\n====================================================== "+Couleurs.VERT+"CHECK CUSTOM IMAGES"+Couleurs.RESET+"\n")
        individualdeployment = individual_deployment()
        check_individual_deployment = individualdeployment.check_if_there_is_custom_images(images_from_dockercompose=images, directories=directories)
        custom_images = "Present" if check_individual_deployment else "Not present"
        print("[ " + Couleurs.VERT + "INDIVIDUAL DEPLOYMENT" + Couleurs.RESET + " ] : ", custom_images)

        ## check mongo replication
        print("\n\n====================================================== "+Couleurs.VERT+"CHECK MONGO REPLICATION"+Couleurs.RESET+"\n")
        mongoanalyzer =  mongo_analyzer()
        mongo_replication = mongoanalyzer.detect_mongo_replication(dockercompose=dockercompose)
        mongo_replication_status = "Present" if mongo_replication is True else  "Not present"
        print("[ " + Couleurs.VERT + "MONGO REPLICATION" + Couleurs.RESET + " ] : ", mongo_replication_status)

        ## check master slave replication
        print("\n\n====================================================== "+Couleurs.VERT+"CHECK MASTER SLAVE REPLICATION"+Couleurs.RESET+"\n")
        masterslave = masterslave_analyzer()
        detect_master_slave_replication = masterslave.detect_master_slave_replication(repository=repository, dockercompose=dockercompose)
        master_slave_replication_status = "Present" if detect_master_slave_replication is True else "Not present"
        print("[ " + Couleurs.VERT + "MASTER SLAVE REPLICATION" + Couleurs.RESET + " ] : ", master_slave_replication_status)

        ## check events
        print("\n\n====================================================== "+Couleurs.VERT+"CHECK EVENTS"+Couleurs.RESET+"\n")
        event_analyse = event_analyser()
        events_check = event_analyse.check_event_sourcing(images)
        events_status = "Present" if events_check is True else "Not present"
        print("[ " + Couleurs.VERT + "CHECK EVENTS" + Couleurs.RESET + " ] : ", events_status)

        ## load balancing and scaling
        print("\n\n====================================================== "+Couleurs.VERT+"LOAD BALANCING AND SCALING"+Couleurs.RESET+"\n")
        lb = loadbalancer_analyzer()
        load_balancing_check = lb.detect_load_balancer(repository=repository, images=images)
        load_balancing_status = lb.process_load_balancer_result(load_balancing_check)
        print("[ " + Couleurs.VERT + "LOAD BALANCING STATUS" + Couleurs.RESET + " ] : ", load_balancing_status)

        ## CI/CD
        print("\n\n====================================================== "+Couleurs.VERT+"CICD ANALYSER"+Couleurs.RESET+"\n")
        cicd = cicd_analyzer()
        check_services_in_CICD_status = cicd.check_services_in_CI(repository=repository, directories=directories)
        print("[ " + Couleurs.VERT + "CI/CD" + Couleurs.RESET + " ] : ", check_services_in_CICD_status)

        ## check gateway
        print("\n\n======================================================= "+Couleurs.VERT+"CHECK GATEWAY"+Couleurs.RESET+"\n")
        gatewayanalyse = gateway_analyzer()
        gateway_check = gatewayanalyse.detect_gateway(dockercompose=dockercompose,directories=directories)
        gateway_status = "Present" if gateway_check is True else "Not present"
        print("[ " + Couleurs.VERT + "GATEWAY" + Couleurs.RESET + " ] : ", gateway_status)

    results_df = results_df._append({
        'Repo Name': repository.full_name,
        'Docker Compose Present': docker_compose_status,
        'Custom Images in Docker Compose': custom_images,
        'MongoDB Replication': mongo_replication_status,
        'Master Slave Replication': master_slave_replication_status,
        'Events': events_status,
        'Microservices in CI/CD': check_services_in_CICD_status,
        'Load Balancing': load_balancing_status, 
        'DBs unique': db_analyser_status,
        'Gateway': gateway_status

    }, ignore_index=True)

    return results_df

def main(token, input_file):
    access_token = token
    print("RETRO ANALYSE")
    g = Github(access_token)

    # Create an empty DataFrame to store results
    columns = [
        'Repo Name', 'Docker Compose Present', 'Custom Images in Docker Compose',
        'MongoDB Replication', 'Master Slave Replication', 'Events',
        'Microservices in CI/CD', 'Load Balancing', 'DBs unique','Gateway'
    ]

    results_df = pd.DataFrame(columns=columns)
    output_file = "./output/output.xlsx"

    columns_validation = ['Repo Name', 'Microservice or not']
    validation_df = pd.DataFrame(columns=columns_validation)
    validation_output_file = "./output/validation.xlsx"


    with open("./extracted_data.csv", "r") as csv_file:
        for line in csv_file:
            repo_name = line.strip()
            repository = g.get_repo(repo_name)
            print(repository)
            try:
                results_df = analyze_repository(repository, results_df, access_token)
                save_to_excel(results_df, output_file)

                print(results_df)
            except Exception as e:
                print("ERROR : ", e)
                continue

    with   open("./extracted_data.csv", "r") as csv_file:
        for line in csv_file:
            repo_name = line.strip()
            repository = g.get_repo(repo_name)
            print(repository)
            try:

                result_validation = microservice_keywords().identify_microservice_keywords(repository)
                validation_df = validation_df._append({
                    'Repo Name': repository.full_name,
                    'Microservice or not': result_validation
                }, ignore_index=True)

                save_to_excel(validation_df, validation_output_file)
                print(validation_df)
            except Exception as e:
                print("ERROR : ", e)
                continue


def save_to_excel(results_df, output_file):
    # Save the results to an Excel file
    results_df.to_excel(output_file, index=False)

    # Adjust column widths
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, index=False, sheet_name='Results')
        worksheet = writer.sheets['Results']
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            adjusted_width = (max_length + 2) * 1.2
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub script with access token and output file arguments")
    parser.add_argument("--token", required=True, help="GitHub access token")
    parser.add_argument("--input", required=True, help="input csv file path")
    args = parser.parse_args()
    main(args.token, args.input)