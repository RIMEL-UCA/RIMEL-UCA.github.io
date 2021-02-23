echo "Raise"

grep -rnw 'sdk\python\kfp\dsl' -e 'raise [A-z]*Error' | wc -l 
# wc -l is optional

echo "Raise Exception"

grep -rnw 'sdk\python\kfp\dsl' -e 'raise [A-z]*Exception' | wc -l
# wc -l is optional

echo "Errors except"

grep -rnw 'sdk/python' -e 'except:' -B 3 -A 3