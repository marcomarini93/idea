#!/bin/bash                                                                                                                                                                                                                        

. ./path.sh
. ./cmd.sh

# ./local/lat2phone.sh exp_2/train/tri3/decode_test/lat.1.gz exp_2/train/tri3/final.mdl exp_2/train/tri3/phones.txt out_2.ctm

lat_file=$1
#folder_file=train/tri3/decode_test/lat.1.gz
model=$2
phones=$3
output_file=$4

if [[ ! -f $lat_file || ! -f $model || ! -f $phones || ! $output_file ]];then
    echo "Usage: $0 <lat_zip_file> <model> <phones> <output_file>"
    echo "<lat_zip_file>:input lat zip file"
    echo "<model>:final.mdl"
    echo "<phones>:phones text file"
    echo "<output_file>:output file"
    exit 1;
else
	#for index in 2 3 4
	#do
		#lat_file=exp_$index/$folder_file
		#lattice-align-phones --replace-output-symbols=true $model ark:"gunzip -c ${lat_file}|" ark:- |lattice-to-ctm-conf ark:- - |utils/int2sym.pl -f\
		5 $phones > ${output_file}_$index
	#done
	lattice-align-phones --replace-output-symbols=true $model ark:"gunzip -c ${lat_file}|" ark:- |lattice-to-ctm-conf ark:- - |utils/int2sym.pl -f\
		5 $phones > $output_file
    echo "done"
fi