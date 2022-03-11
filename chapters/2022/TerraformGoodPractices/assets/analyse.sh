if [ "$1" == "--help" ]
  then
    echo "Usage: sh $0 $1 [INPUT_FILE] [OUTPUT_FILE]"
    echo "Analyse all github repository listed in the input file and summurize them in a markdown file."
    echo "default input: input.txt"
    echo "default ouput: result.md"
	exit 0
fi

if [ -z "$1" ]
  then
    INPUT_FILE="input.txt"
  else
    INPUT_FILE=$1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: $INPUT_FILE do not exist." >&2
	exit -1
fi

if [ -z "$2" ]
  then
    OUTPUT_FILE="result.md"
  else
    OUTPUT_FILE=$2
fi

readarray -t ARRAY <"${INPUT_FILE}"
echo "| Projet | Taux de succès |" > "${OUTPUT_FILE}"
echo "| :--------- | :-----------------------: |" >> "${OUTPUT_FILE}"
$(for project in ${ARRAY[@]}
do
	echo "Scanning ${project}..." | tr -d '\r' >&2
	git clone "$project" tmp_project 2> poubelle
	checkov -d tmp_project --compact --framework terraform --check CKV_AWS_41,CKV_AWS_45,CKV_AWS_46,CKV_AWS_58,CKV_AWS_149,CKV_AZURE_41,CKV_BCW_1,CKV_GIT_4,CKV_LIN_1,CKV_OCI_1,CKV_OPENSTACK_1,CKV_PAN_1 | grep -w 'Passed checks' | egrep -o '[0-9]*' > checkov_result
	VALUE_COUNT=$(cat checkov_result | wc -l)
	if [ "$VALUE_COUNT" -lt "2" ]; then
		echo "| ${project} | NULL |" | tr -d '\r' >> "${OUTPUT_FILE}"
	else
		VALUE_PASSED=$(cat checkov_result | head -n 1)
		VALUE_FAILED=$(cat checkov_result | head -n 2 | tail -1)
		VALUE_TOTAL=$((${VALUE_PASSED}+${VALUE_FAILED}))
		if [ "$VALUE_TOTAL" == "0" ]; then
			echo "| ${project} | 100% |" | tr -d '\r' >> "${OUTPUT_FILE}"
		else
			echo "| ${project} | $((${VALUE_PASSED}*100/${VALUE_TOTAL}))% |" | tr -d '\r' >> "${OUTPUT_FILE}"
		fi
	fi
	rm -rf tmp_project
done)
rm poubelle
rm checkov_result
