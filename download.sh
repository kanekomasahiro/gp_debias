download=download
word_emb=embeddings
glove=$word_emb/glove.txt
gn=$word_emb/gn.txt

mkdir -p $download
mkdir -p $word_emb

git clone https://github.com/uclanlp/gn_glove.git $download
paste download/wordlist/female_word_file.txt download/wordlist/male_word_file.txt > wordlist/gender_pair.tsv 

cp -r download/SemBias ./


get_google_drive () {
    query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=$1" \
        | perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
        | sed -e 's/amp;//g' | sed -n 2p`
    url="https://drive.google.com$query"
    curl -b ./cookie.txt -L -o $2 $url
}

file_id="1jrbQmpB5ZNH4w54yujeAvNFAfVEG0SuE"
filename="embeddings/glove.zip"
get_google_drive $file_id $filename
unzip $filename 
rm $filename 
sed -i "1s/^/322636 300\n/" vectors.txt
mv vectors.txt $glove


file_id="1v82WF43w-lE-vpZd0JC1K8WYZQkTy_ii"
filename="embeddings/gn.zip"
get_google_drive $file_id $filename

unzip $filename
rm $filename 
sed -i "1s/^/322636 300\n/" vectors300.txt
mv vectors300.txt $gn

rm ./cookie.txt
rm -r __MACOSX
yes | rm -r $download
