save_dir=src/debiased_$1
debiased_dir=debiased_embeddings

mkdir -p src/$1\_model
mkdir -p src/debiased_$1

python src/train.py $1
python src/eval.py $1

mkdir -p $debiased_dir
mv $save_dir/gender_debiased.bin $debiased_dir/gp-$1\.bin
mv $save_dir/gender_debiased.txt $debiased_dir/gp-$1\.txt
rm -r src/debiased_$1
