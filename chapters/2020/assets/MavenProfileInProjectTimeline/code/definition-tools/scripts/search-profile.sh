debug=false

# Check args
for arg in "$@"
do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]
    then
        echo "search-profile [Result Directory] [Profile name pattern] [...Options]"
        echo "Search specific regexp profile matching pattern in results csv"
        echo ""
        echo "Available options:"
        printf '\t-d --debug    Generate the file where the search is done.\n\n'
        printf '\t-h --help     Help.\n\n'
        exit 0
    elif [ "$arg" == "--debug" ] || [ "$arg" == "-d" ]
    then
        find $1 -type f -name '*.csv' -exec cat {} \; | sort | uniq | cut -d ';' -f3- | grep '\S' > DEBUG-SEARCH-PROFILE.csv
        echo "Check out DEBUG-SEARCH-PROFILE.csv to see profiles"
    fi
done

# Search
TOTAL=$(find $1 -type f -name '*.csv' -exec cat {} \; | sort | uniq | cut -d ';' -f3- | grep '\S' | wc -l)
FIND=$(find $1 -type f -name '*.csv' -exec cat {} \; | sort | uniq | cut -d ';' -f3- | grep '\S' | grep -E $2 -c)
echo "Found : $FIND / $TOTAL"


