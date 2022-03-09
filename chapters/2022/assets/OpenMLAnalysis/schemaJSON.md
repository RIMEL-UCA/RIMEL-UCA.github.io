# Documentation Schema JSON

Cette courte documentation explique comment interpréter les différents JSON disponibles dans le fichier [Sauvegarde des données d'expérimentation avec résultats brutes.zip](./Sauvegarde%20des%20donn%C3%A9es%20d'exp%C3%A9rimentation%20avec%20r%C3%A9sultats%20brutes.zip).

A noter pour les futures explications que `index DF` correspond à `index DataFrame`, c’est une valeur utilisée seulement pour la transaction de nos données vers le format JSON et inversement. 

## Résultat 

### Occurrence Flow

#### Fichiers concernés : 
- `result_sorted_flow_occurence_All_Times_Series.json`
- `result_sorted_flow_occurence_all_not_time_series.json`

#### Schema

```jsonc
{ // The root schema
    "0": { // occurrence root : name index DF
        "0": 1090, // flow id
        "1": 113 // nombre d'utilisations
    },
    "1": { // occurrence root : name index DF
        "0": 1068, // flow id
        "1": 98 // nombre d'utilisations
    },
    "2": { // occurrence root : name index DF
        "0": 377, // flow id
        "1": 55 // nombre d'utilisations
    }
}
```

### Comparaison Occurrence Flow

#### Fichiers concernés : 
- `result_comparaison_flow_occurrence.json`

#### Schema

```jsonc
{ // The root schema
    "1090": { // comparaison root, name : flow id
        "tm": 113, // occurrence in dataset time series
        "other": 50 // occurrence in dataset that are not time series
    },
    "1068": { // comparaison root, name : flow id
        "tm": 98, // occurrence in dataset time series
        "other": 42 // occurrence in dataset that are not time series
    },
    "377": { // comparaison root, name : flow id
        "tm": 55, // occurrence in dataset time series
        "other": 20 // occurrence in dataset that are not time series 
    }
}
```

### Flow Tree

#### Fichiers concernés : 
- `result_flows_treeJapaneseVowels.json`
- `result_flows_tree_All_Times_Series.json`
- `result_unique_flow_tree_All_Times_Series.json`
- `result_flow_tree_all_not_time_series.json`

#### Schema

```jsonc
{ // The root schema
    "0": { // index DF
        "0": 56 // flow id (here 56 is the id)
    },
    "75": { // index DF
        "0": { // index DF
            "133": [ // flow id 
                60, // component of (133): flow id
                134, // component of (133): flow id
                135 // component of (133): flow id
            ]
        }
    },
    "664": { // index DF
        "0": { // index DF
            "8786": [ // flow id 
                {
                    "8775": [ // component of (8786): flow id
                        {
                            "8776": [ // component of (8775): flow id
                                8777, // component of (8776): flow id
                                8778, // component of (8776): flow id
                                8779 // component of (8776): flow id
                            ]
                        },
                        {
                            "8780": [ // component of (8775): flow id
                                8781, // component of (8780): flow id
                                8782 // component of (8780): flow id
                            ]
                        }
                    ]
                },
                8790 // component of (8786): flow id
            ]
        }
    }
}
```

## Liste de Tasks

Représente la liste des tâches utilisées pour réaliser l'expérience. La représentation utilisée pour représenter une tâche est celle de l’**API Python d'OpenML**. [OpenML task](https://openml.github.io/openml-python/develop/api.html#module-openml.tasks)

### Fichiers concernés : 
- `tasks_JapaneseVowels.json`
- `tasks_all_times_series.json`
- `tasks_all_not_time_series.json`

### Schema 

```jsonc
{ // The root schema
    "3510": { // task root, name : task id
        "tid": 3510, // task id
        "ttid": {
            "name": "SUPERVISED_CLASSIFICATION" // task type
        },
        "did": 375, // dataset id
        "name": "JapaneseVowels", // dataset name
        "task_type": "Supervised Classification", // task type
        "status": "active",
        "estimation_procedure": "10-fold Crossvalidation",
        "source_data": "375",
        "target_feature": "speaker",
        "MajorityClassSize": 1614.0,
        "MaxNominalAttDistinctValues": 9,
        "MinorityClassSize": 782.0,
        "NumberOfClasses": 9,
        "NumberOfFeatures": 15,
        "NumberOfInstances": 9961,
        "NumberOfInstancesWithMissingValues": 0,
        "NumberOfMissingValues": 0,
        "NumberOfNumericFeatures": 14,
        "NumberOfSymbolicFeatures": 1,
        "evaluation_measures": null,
        "quality_measure": null,
        "target_value": null,
        "number_samples": null
    }
}
```
## Liste de Runs

Représente la liste des runs utilisées pour réaliser l'expérience. La représentation utilisée pour représenter une run est celle de l’**API Python d'OpenML**. [OpenML run](https://openml.github.io/openml-python/develop/api.html#module-openml.runs)

### Fichiers concernés : 
- `runs_JapaneseVowels.json`
- `runs_all_times_series.json`
- `runs_all_not_time_series.json`

### Schema 

```jsonc
{ // The root Schema
    "12": { // run root, name : run id
        "run_id": 12, // run id
        "task_id": 39, // task id
        "setup_id": 2,
        "flow_id": 57, // flow id
        "uploader": 1,
        "task_type": {
            "name": "SUPERVISED_CLASSIFICATION" // task type
        },
        "upload_time": "2014-04-06 23:42:09",
        "error_message": ""
    }
}
```
## Liste de Flows

Représente la liste des flows utilisées pour réaliser l'expérience. La représentation utilisée pour représenter une tâche est celle de l’**API Python d'OpenML**. [OpenML flow](https://openml.github.io/openml-python/develop/api.html#module-openml.flows)

### Fichiers concernés : 
- `flows_JapaneseVowels.json`
- `flows_All_Times_Series.json`
- `flows_all_not_time_series.json`

### Schema 

```jsonc
{ // The root schema
    "29": { // index DF
        "0": { // root flow, name : index DF
            "binary_format": null,
            "binary_url": null,
            "components": { // Component list of this flow, each flow is completely defined.
                "K": { // root component flow
                    "binary_format": null,
                    "binary_url": null,
                    "components": {},
                    "dependencies": "Weka_3.7.10",
                    "external_version": "Weka_3.7.10_8034",
                    "id": 71, // component flow id
                    "model": null,
                    "openml_url": "https:\/\/www.openml.org\/f\/71",
                    "parameters_meta_info": {
                        "C": {
                            "description": "The size of the cache (a prime number), 0 for full cache and \n\t-1 to turn it off.\n\t(default: 250007)",
                            "data_type": "option"
                        },
                        "D": {
                            "description": "Enables debugging output (if available) to be printed.\n\t(default: off)",
                            "data_type": "flag"
                        },
                        "E": {
                            "description": "The Exponent to use.\n\t(default: 1.0)",
                            "data_type": "option"
                        },
                        "L": {
                            "description": "Use lower-order terms.\n\t(default: no)",
                            "data_type": "flag"
                        },
                        "no-checks": {
                            "description": "Turns off all checks - use with caution!\n\t(default: checks on)",
                            "data_type": "flag"
                        }
                    },
                    "tags": [],
                    "upload_date": "2014-04-04T14:39:43",
                    "version": "1"
                }
            },
            "dependencies": "Weka_3.7.10",
            "external_version": "Weka_3.7.10_8034",
            "id": 70, // flow id
            "model": null,
            "openml_url": "https:\/\/www.openml.org\/f\/70",
            "parameters_meta_info": {
                "C": {
                    "description": "The complexity constant C. (default 1)",
                    "data_type": "option"
                },
                "D": {
                    "description": "If set, classifier is run in debug mode and\n\tmay output additional info to the console",
                    "data_type": "flag"
                },
                "E": {
                    "description": "The Exponent to use.\n\t(default: 1.0)",
                    "data_type": "option"
                },
                "K": {
                    "description": "The Kernel to use.\n\t(default: weka.classifiers.functions.supportVector.PolyKernel)",
                    "data_type": "kernel"
                },
                "L": {
                    "description": "The tolerance parameter. (default 1.0e-3)",
                    "data_type": "option"
                },
                "M": {
                    "description": "Fit logistic models to SVM outputs.",
                    "data_type": "flag"
                },
                "N": {
                    "description": "Whether to 0=normalize\/1=standardize\/2=neither. (default 0=normalize)",
                    "data_type": "option"
                },
                "P": {
                    "description": "The epsilon for round-off error. (default 1.0e-12)",
                    "data_type": "option"
                },
                "V": {
                    "description": "The number of folds for the internal\n\tcross-validation. (default -1, use training data)",
                    "data_type": "option"
                },
                "W": {
                    "description": "The random number seed. (default 1)",
                    "data_type": "option"
                },
                "no-checks": {
                    "description": "Turns off all checks - use with caution!\n\tTurning them off assumes that data is purely numeric, doesn't\n\tcontain any missing values, and has a nominal class. Turning them\n\toff also means that no header information will be stored if the\n\tmachine is linear. Finally, it also assumes that no instance has\n\ta weight equal to 0.\n\t(default: checks on)",
                    "data_type": "flag"
                }
            },
            "tags": [
                "Verified_Learning_Curve,Verified_Supervised_Classification"
            ],
            "upload_date": "2014-04-04T14:39:43",
            "version": "1"
        }
    }
}
```
