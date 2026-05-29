cd ./..
for i in {29..31}
do
    sleep 5
    python teleoperation.py --id $i 
done
